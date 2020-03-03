import numpy as np
import pyccl as ccl
import HOD_theory as hod
import SZ_theory as sz


class GSKY_Theory:

    # Wavenumbers and scale factors
    k_arr = np.geomspace(1E-4,1E1,256)
    a_arr = np.linspace(0.1,1,32)

    def __init__ (self, Nz):
        """ Nz -- list of (zarr,Nzarr) """

        self.Nz = Nz
        self.params = {
            'mmin'  : 12.02,
            'mminp' : -1.34,
            'm0'     : 6.6,
            'm0p'    : -1.43,
            'm1'     : 13.27, 
            'm1p'    : 0.323,
            'bhydro' : 1
            }
        self.paramnames=self.params.keys()
        self.C = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)
        # We will use a mass definition with Delta = 200 times the matter density
        self.hmd_200m = ccl.halos.MassDef200m()
        # The Duffy 2008 concentration-mass relation
        self.cM = ccl.halos.ConcentrationDuffy08(self.hmd_200m)
        # The NFW profile to characterize the matter density around halos
        self.pM = ccl.halos.profiles.HaloProfileNFW(self.cM)
        self.have_spectra=False
        self._setup_Cosmo()
        self._setup_HOD()
        
    def set_params(self,params):
        for k in params.keys():
            if k not in self.paramnames:
                print ("Warning, parameter %s not recognized."%k)
                stop()
        self.params.update(params)
        self.have_spectra=False
        self._setup_HOD()
        
    def set_cosmology(self,C):
        self.C = C
        self.have_spectra=False
        self._setup_Cosmo()
        self._setup_HOD()
        
    def _setup_Cosmo(self):
        # Now we can put together HMCalculator
        # The Tinker 2008 mass function
        self.nM = ccl.halos.MassFuncTinker08(self.C, mass_def=self.hmd_200m)
        # The Tinker 2010 halo bias
        self.bM = ccl.halos.HaloBiasTinker10(self.C, mass_def=self.hmd_200m)
        self.hmc = ccl.halos.HMCalculator(self.C, self.nM, self.bM, self.hmd_200m)
        
    def _setup_HOD(self):
        # HOD auto
        self.HOD2pt = hod.Profile2ptHOD()
        p=self.params
        self.pg = hod.HaloProfileHOD(c_M_relation=self.cM,
                                     lMmin=p['mmin'], lMminp=p['mminp'],
                                     lM0=p['m0'], lM0p=p['m0p'],
                                     lM1=p['m1'], lM1p=p['m1p'])
        self.py = sz.HaloProfileArnaud(b_hydro=p['bhydro'])

        ## now tracers
        self.tg = [ccl.NumberCountsTracer(self.C, False, (z_arr, nz_arr),
                             bias=(z_arr, np.ones_like(z_arr))) for
                 z_arr,nz_arr in self.Nz]
        self.ts = [ccl.WeakLensingTracer(self.C, (z_arr, nz_arr)) for
                   z_arr,nz_arr in self.Nz]
        self.tk = ccl.CMBLensingTracer(self.C,z_source=1150)
        self.ty = sz.SZTracer(self.C)

    def _get_power_spectra(self):
        if not self.have_spectra:
            self.pk_MMf = ccl.halos.halomod_Pk2D(self.C, self.hmc, self.pM,
                                normprof1=True,
                                lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
            self.pk_gMf = ccl.halos.halomod_Pk2D(self.C, self.hmc, self.pg, prof2=self.pM,
                                normprof1=True, normprof2=True,
                                lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
            self.pk_ggf = ccl.halos.halomod_Pk2D(self.C, self.hmc, self.pg, prof_2pt=self.HOD2pt,
                                normprof1=True,
                                lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
            self.have_spectra=True
    def _tracer(self, letter, i):
        if letter == 'g':
            return self.tg[i]
        elif letter == 's':
            return self.ts[i]
        elif letter == 'k':
            return self.tk
        elif letter == 'y':
            return self.ty
        else:
            print ("Tracer %s not recognized."%(letter))
            stop()
        
    def getCls (self,typ, l_arr, i=0, j=0):
        """ typ - is a two character string gg, gs,ss, sy, sk etc...
            i,j are indices for g and s"""

        self._get_power_spectra()
        tracer1 = self._tracer(typ[0],i)
        tracer2 = self._tracer(typ[1],j)
        if typ == "gg":
            Pk = self.pk_ggf
        elif typ=="gs" or type=="sg":
            Pk = self.pk_gMf
        else:
            Pk = self.pk_MMf

        return ccl.angular_cl(self.C, tracer1, tracer2, l_arr, p_of_k_a=Pk)
            
