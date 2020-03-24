import pyccl as ccl
import numpy as np
from pyccl.halos.concentration import Concentration
from pyccl.halos.massdef import MassDef

class ConcentrationDuffy08M500(Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.
    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08M500'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08M500, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(500, 'critical')

    def _check_mdef(self, mdef):
        if int(mdef.Delta) != 500 or mdef.rho_type != 'critical':
            return True
        return False

    def _setup(self):
        self.A = 3.67
        self.B = -0.0903
        self.C = -0.51

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)


class HaloProfileArnaud(ccl.halos.HaloProfile):
    def __init__(self, b_hydro, rrange=(1e-3, 10), qpoints=100):
        self.c500 = 1.81
        self.alpha = 1.33
        self.beta = 4.13
        self.gamma = 0.31
        self.rrange = rrange
        self.qpoints = qpoints
        self.b_hydro = b_hydro

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
        P0 = 6.41  # reference pressure

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
        R = mass_def.get_radius(cosmo, M_use, a) / a

        nn = self._norm(cosmo, M_use, a, b)
        prof = self._form_factor(r_use[None, :] * R[:, None])
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
        R = mass_def.get_radius(cosmo, M_use, a) / a

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

        # eV2J = 1.6e-19 eV/J (J=kg m2/s2)
        # cm2pc = 3.1e18 cm/pc

        # prefac = (sigma_t*(10**2)**2/(m_e*c**2/J2eV))*cm2pc*10**6

        prefac = 4.017100792437957e-06
        w_arr = prefac * a_arr

        self._trc = []
        self.add_tracer(cosmo, kernel=(chi_arr, w_arr))
