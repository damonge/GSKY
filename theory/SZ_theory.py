import pyccl as ccl
import numpy as np
import copy
import scipy.interpolate

G_MPC_MSUN = 4.5171e-48 # MPc^3/MSun/s^2 (6.67408e-11*(3.085677581491367399198952281E+22)**-3*1.9884754153381438E+30)

class HaloProfileBattaglia(ccl.halos.HaloProfile):
    def __init__(self):
        self.M_PIV = 1e14 # MSun
        self.alpha = 1.
        self.gamma = -0.3

        # Battaglia et al., 2012, Tab. 1, AGN feedback, Delta = 200
        # P0
        self.A_P0 = 18.1
        self.alpm_P0 = 0.154
        self.alpz_P0 = -0.758
        # beta
        self.A_bt = 4.35
        self.alpm_bt = 0.0393
        self.alpz_bt = 0.415
        # xc
        self.A_xc = 0.497
        self.alpm_xc = -0.00865
        self.alpz_xc = 0.731

        # Battaglia et al., 2012, Tab. 1, AGN feedback, Delta = 500
        # P0
        # self.A_P0 = 7.49
        # self.alpm_P0 = 0.226
        # self.alpz_P0 = -0.957
        # # beta
        # self.A_bt = 4.19
        # self.alpm_bt = 0.0480
        # self.alpz_bt = 0.615
        # # xc
        # self.A_xc = 0.710
        # self.alpm_xc = -0.0833
        # self.alpz_xc = 0.853

        self.xarr = np.logspace(-4, 5, 200)

        super(HaloProfileBattaglia, self).__init__()

    def _P0(self, M, a):
        """
        P0, Eq. (11) in Battaglia et al., 2012
        :param M:
        :param a:
        :return:
        """

        P0 = self.A_P0*(M/self.M_PIV)**self.alpm_P0*(1./a)**self.alpz_P0

        return P0

    def _beta(self, M, a):
        """
        beta, Eq. (11) in Battaglia et al., 2012
        :param M:
        :param a:
        :return:
        """

        beta = self.A_bt * (M / self.M_PIV) ** self.alpm_bt * (1. / a) ** self.alpz_bt

        return beta

    def _xc(self, M, a):
        """
        xc, Eq. (11) in Battaglia et al., 2012
        :param M:
        :param a:
        :return:
        """

        xc = self.A_xc * (M / self.M_PIV) ** self.alpm_xc * (1. / a) ** self.alpz_xc

        return xc

    def _form_factor(self, x, M, a):
        """
        Eq. (10) in Battaglia et al., 2012 (defined in physical units)
        :param x: x = r_phys/R200c,phys
        :param M:
        :param a:
        :return:
        """

        if M.shape[0] > 1:
            M_use = M[:, np.newaxis]
            if x.ndim == 1:
                x_use = x[np.newaxis, :]
            else:
                x_use = copy.deepcopy(x)
        else:
            M_use = copy.deepcopy(M)
            x_use = copy.deepcopy(x)

        P0 = self._P0(M_use, a)
        beta = self._beta(M_use, a)
        xc = self._xc(M_use, a)

        x_o_xc = x_use/xc

        f1 = P0*x_o_xc**self.gamma
        f2 = (1 + x_o_xc**self.alpha)**(-beta)

        f = f1*f2

        return f

    def _sinc_interp(self, x):

        if not hasattr(self, 'sinc_interp'):
            x_interp = np.logspace(-9, 6, 1000)
            sinc = np.sin(x_interp)/x_interp
            self.sinc_interp = scipy.interpolate.UnivariateSpline(x_interp, sinc)

        return self.sinc_interp(x)

    def _fourier_integ(self, kR, M, a):
        """
        u(k|M) = int_0^inf dx d sin(k_com R200c,com x)/(k_com R200c,com) P_e(x|M)
        x = r_phys/R200c,phys
        :param kR: k_com R200c,com
        :param M:
        :param a:
        :return:
        """

        ff = self._form_factor(self.xarr, M, a)
        if ff.ndim > 1:
            if not hasattr(self, 'xuse'):
                self.xuse = self.xarr[np.newaxis, np.newaxis, :]
            kR_use = kR[:, :, np.newaxis]
            ff = ff[:, np.newaxis, :]
        else:
            if not hasattr(self, 'xuse'):
                self.xuse = copy.deepcopy(self.xarr)
            kR_use = copy.deepcopy(kR)

        integ = self.xuse**2*self._sinc_interp(kR_use*self.xuse)*ff

        fourier_prof = np.trapz(integ, self.xuse, axis=-1)

        return fourier_prof


    def _norm(self, cosmo, mass_def, M, a):
        """
        Battaglia profile normalization (P200, above Eq. (9) in Battaglia et al., 2012)
        N = G M200c 200 rho_crit,phys(a) f_b/(2 R200c,phys)
        f_b = Omega_b/Omega_m
        Units are Msun/(Mpc s^2)
        :param cosmo:
        :param mass_def:
        :param M:
        :param a:
        :return:
        """

        # Since the Battaglia profile is defined in physical units the norm needs to be in physical units as well
        R_Delta = mass_def.get_radius(cosmo, M, a)
        f_b = cosmo['Omega_b']/cosmo['Omega_m']

        P_Delta = G_MPC_MSUN*M*mass_def.get_Delta(cosmo, a)*ccl.rho_x(cosmo, a, 'critical')*f_b/(2.*R_Delta)

        return P_Delta

    def _real(self, cosmo, r, M, a, mass_def):
        """
        Real space pressure profile.
        :param cosmo:
        :param r: comoving
        :param M:
        :param a:
        :param mass_def:
        :return:
        """

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Since r is comoving we need comoving R_Delta radius
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use, a)/a

        x = r_use[np.newaxis, :]/R[:, np.newaxis]

        nn = self._norm(cosmo, mass_def, M_use, a)
        prof = self._form_factor(x, M_use, a)

        prof *= nn[:, None]

        # P_th = 5(X_H+3)/(2(X_H+1)) P_e, X_H = 0.76
        P_e_frac = 1.932
        prof /= P_e_frac

        # Change units of normalization
        # P_Delta [kg/(m s^2)] = P_Delta [Msun/(Mpc s^2)] * Msun_to_kg/Mpc_to_m^3
        # P_Delta [kg/(m s^2)] = P_Delta [kg m^2/(m^3 s^2)] = P_Delta [J/(m^3)]
        # P_Delta [eV/(m^3)] = P_Delta [J/(m^3)] * Joules_to_eV
        # P_Delta [eV/(m^3)] = # P_Delta [eV/(cm^3)] /(m_to_cm^3)
        # Msun_to_kg = 1.9889e30
        # Mpc_to_m = 3.0857e22
        # Joules_to_eV = 6.24e18
        # m_to_cm = 100
        norm_transf = 4.022602961046903e+20
        prof *= norm_transf

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        """
        2D Fourier transform of Battaglia profile
        Units: eV/cm^3 Mpc^3 (comoving)
        :param cosmo:
        :param k: comoving
        :param M:
        :param a:
        :param mass_def:
        :return:
        """
        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving R_Delta radius
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use, a)/a

        ff = self._fourier_integ(k_use[None, :] * R[:, None], M_use, a)
        nn = self._norm(cosmo, mass_def, M_use, a)

        prof = (4*np.pi*R**3 * nn)[:, None] * ff

        # P_th = 5(X_H+3)/(2(X_H+1)) P_e, X_H = 0.76
        P_e_frac = 1.932
        prof /= P_e_frac

        # Change units of normalization
        # P_Delta [kg/(m s^2)] = P_Delta [Msun/(Mpc s^2)] * Msun_to_kg/Mpc_to_m^3
        # P_Delta [kg/(m s^2)] = P_Delta [kg m^2/(m^3 s^2)] = P_Delta [J/(m^3)]
        # P_Delta [eV/(m^3)] = P_Delta [J/(m^3)] * Joules_to_eV
        # P_Delta [eV/(m^3)] = # P_Delta [eV/(cm^3)] /(m_to_cm^3)
        # Msun_to_kg = 1.9889e30
        # Mpc_to_m = 3.0857e22
        # Joules_to_eV = 6.24e18
        # m_to_cm = 100
        norm_transf = 4.022602961046903e+20
        prof *= norm_transf

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class HaloProfileArnaud(ccl.halos.HaloProfile):
    def __init__(self, b_hydro, rrange=(1e-3, 10), qpoints=100):
        # Values from Planck  2013 (Planck intermediate results: V.Pressure profiles of galaxy clusters from
        # the Sunyaev - Zeldovich effect
        self.c500 = 1.81
        self.alpha = 1.33
        self.beta = 4.13
        self.gamma = 0.31
        self.rrange = rrange
        self.qpoints = qpoints
        self.b_hydro = b_hydro

        # Values from Arnaud et al., 2010
        # self.c500 = 1.177
        # self.alpha = 1.051
        # self.beta = 5.4905
        # self.gamma = 0.3081

        # Interpolator for dimensionless Fourier-space profile
        self._fourier_interp = self._integ_interp()
        super(HaloProfileArnaud, self).__init__()

    def _update_bhydro(self, b_hydro):
        self.b_hydro = b_hydro

    def _form_factor(self, x):
        f1 = (self.c500*x)**(-self.gamma)
        f2 = (1+(self.c500*x)**self.alpha)**(-(self.beta-self.gamma)/self.alpha)
        return f1*f2

    def _integ_interp(self):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from numpy.linalg import lstsq

        def integrand(x):
            return self._form_factor(x)*x

        # # Integration Boundaries # #
        rmin, rmax = self.rrange
        lgqmin, lgqmax = np.log10(1/rmax), np.log10(1/rmin)  # log10 bounds

        q_arr = np.logspace(lgqmin, lgqmax, self.qpoints)
        f_arr = np.array([quad(integrand,
                               a=1e-4, b=np.inf,     # limits of integration
                               weight="sin",  # fourier sine weight
                               wvar=q)[0] / q
                          for q in q_arr])

        F2 = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic")

        # # Extrapolation # #
        # Backward Extrapolation
        def F1(x):
            if np.ndim(x) == 0:
                return f_arr[0]
            else:
                return f_arr[0] * np.ones_like(x)  # constant value

        # Forward Extrapolation
        # linear fitting
        Q = np.log10(q_arr[q_arr > 1e2])
        F = np.log10(f_arr[q_arr > 1e2])
        A = np.vstack([Q, np.ones(len(Q))]).T
        m, c = lstsq(A, F, rcond=None)[0]

        def F3(x):
            return 10**(m*x+c)  # logarithmic drop

        def F(x):
            return np.piecewise(x,
                                [x < lgqmin,        # backward extrapolation
                                 (lgqmin <= x)*(x <= lgqmax),  # common range
                                 lgqmax < x],       # forward extrapolation
                                [F1, F2, F3])
        return F

    def _norm(self, cosmo, M, a, b):
        """Computes the normalisation factor of the Arnaud profile.
        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
        (Arnaud et al., 2009)
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        # Value from Planck  2013 (Planck intermediate results: V.Pressure profiles of galaxy clusters from
        # the Sunyaev - Zeldovich effect
        P0 = 6.41  # reference pressure
        # Values from Arnaud et al., 2010
        # P0 = 8.403*h70**(-3./2)

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        PM = (M*(1-b))**(2/3+aP)             # mass dependence
        Pz = ccl.h_over_h0(cosmo, a)**(8/3)  # scale factor (z) dependence

        P = K * PM * Pz
        return P

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        # hydrostatic bias
        b = self.b_hydro
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use * (1-b), a) / a

        nn = self._norm(cosmo, M_use, a, b)
        prof = self._form_factor(r_use[None, :] / R[:, None])
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        """Computes the Fourier transform of the Arnaud profile.
        .. note:: Output units are ``[norm] Mpc^3``
        """
        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # hydrostatic bias
        b = self.b_hydro
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use * (1-b), a) / a

        ff = self._fourier_interp(np.log10(k_use[None, :] * R[:, None]))
        nn = self._norm(cosmo, M_use, a, b)

        prof = (4*np.pi*R**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class SZTracer(ccl.Tracer):
    def __init__(self, cosmo, z_max=6., n_chi=1024):
        self.chi_max = ccl.comoving_radial_distance(cosmo, 1./(1+z_max))
        chi_arr = np.linspace(0, self.chi_max, n_chi)
        a_arr = ccl.scale_factor_of_chi(cosmo, chi_arr) 
        # avoid recomputing every time
        # Units of eV * Mpc / cm^3

        # sigma_T = 6.65e-29 m2
        # m_e = 9.11e-31 kg
        # c = 3e8  m/s

        # eV_to_Joules = 1.6e-19 eV/J (J=kg m2/s2)
        # cm_to_pc = 3.1e18 cm/pc

        # prefac = (sigma_t*(10**2)**2/(m_e*c**2/eV_to_Joules))*cm_to_pc*10**6

        prefac = 4.01710079e-06
        w_arr = prefac * a_arr

        self._trc = []
        self.add_tracer(cosmo, kernel=(chi_arr, w_arr))
