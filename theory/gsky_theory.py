import numpy as np
import pyccl as ccl
import HOD_theory as hod
import SZ_theory as sz

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)
logger.propagate = False

class GSKYTheory:

    # Wavenumbers and scale factors
    k_arr = np.geomspace(1E-4,1E1,256)
    a_arr = np.linspace(0.1,1,32)

    def __init__ (self, saccfile, params=None):
        """ Nz -- list of (zarr,Nzarr) """

        if params is not None:
            self.params = params
        else:
            self.params = {
                'HODmod': 'zevol',
                'mmin'  : 12.02,
                'mminp' : -1.34,
                'm0'     : 6.6,
                'm0p'    : -1.43,
                'm1'     : 13.27,
                'm1p'    : 0.323,
                'bhydro' : 1
                }

        self.paramnames=self.params.keys()
        self.cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)

        # Setup tracers
        tracer_list = list(saccfile.tracers.values())
        self.tracer_list = tracer_list

        # We will use a mass definition with Delta = 200 times the matter density
        self.hmd_200m = ccl.halos.MassDef200m()
        # The Duffy 2008 concentration-mass relation
        self.cM = ccl.halos.ConcentrationDuffy08(self.hmd_200m)

        self.have_spectra=False
        self._setup_Cosmo()
        self._setup_HM()

    def set_params(self,params):

        logger.info('Setting parameters.')

        for k in params.keys():
            if k not in self.paramnames:
                raise RuntimeError('Parameter {} not recognized. Aborting.'.format(k))
        self.params.update(params)
        self._setup_HM()
        
    def set_cosmology(self, cosmo):

        logger.info('Setting cosmology.')

        self.cosmo = cosmo
        self.have_spectra=False
        self._setup_Cosmo()
        self._setup_HM()
        
    def _setup_Cosmo(self):

        logger.info('Setting up cosmological quantities.')

        # Now we can put together HMCalculator
        # The Tinker 2008 mass function
        self.nM = ccl.halos.MassFuncTinker08(self.cosmo, mass_def=self.hmd_200m)
        # The Tinker 2010 halo bias
        self.bM = ccl.halos.HaloBiasTinker10(self.cosmo, mass_def=self.hmd_200m)

        self.hmc = ccl.halos.HMCalculator(self.cosmo, self.nM, self.bM, self.hmd_200m)

    def _setup_HM(self):

        logger.info('Setting up halo model.')

        self._setup_profiles()
        self._setup_tracers()

    def _setup_profiles(self):

        logger.info('Setting up halo profiles.')

        self.tracer_quantities = [tr.quantity for tr in self.tracer_list]
        if 'cosmic_shear' in self.tracer_quantities or 'kappa' in self.tracer_quantities:
            self.pM = ccl.halos.profiles.HaloProfileNFW(self.cM)
        if 'y' in self.tracer_quantities:
            self.py = sz.HaloProfileArnaud(b_hydro=self.params['bhydro'])
        if 'g' in self.tracer_quantities:
            self.HOD2pt = hod.Profile2ptHOD()
            if self.params['HODmod'] == 'zevol':
                self.pg = hod.HaloProfileHOD(c_M_relation=self.cM,
                                        lMmin=self.params['mmin'], lMminp=self.params['mminp'],
                                        lM0=self.params['m0'], lM0p=self.params['m0p'],
                                        lM1=self.params['m1'], lM1p=self.params['m1p'])
        
    def _setup_tracers(self):

        logger.info('Setting up tracers.')

        p = self.params

        ccl_tracer_dict = {}

        for i, tracer in enumerate(self.tracer_list):
            if tracer.quantity == 'delta_g':
                split_name = tracer.name.split('_')
                if len(split_name) == 2:
                    tracer_no = split_name[1]
                    bias_tup = (p['bz_{}'.format(tracer_no)], p['bb_{}'.format(tracer_no)])
                else:
                    bias_tup = (p['bz'], p['bb'])
                if p['HODmod'] == 'zevol':
                    ccl_tracer_dict[tracer.name] = (ccl.NumberCountsTracer(self.cosmo, False, (tracer.z, tracer.nz),
                                            bias=bias_tup),
                                            self.pg)
                else:
                    ccl_tracer_dict[tracer.name] = (ccl.NumberCountsTracer(self.cosmo, False, (tracer.z, tracer.nz),
                                                                           bias=bias_tup),
                                                    hod.HaloProfileHOD(c_M_relation=self.cM,
                                                                       lMmin=p['mmin'], lMminp=p['mminp'],
                                                                       lM0=p['m0'], lM0p=p['m0p'],
                                                                       lM1=p['m1'], lM1p=p['m1p']))
            elif tracer.quantity == 'Compton_y':
                ccl_tracer_dict[tracer.name] = (sz.SZTracer(self.cosmo),
                                      self.py)
            elif tracer.quantity == 'kappa':
                ccl_tracer_dict[tracer.name] = (ccl.CMBLensingTracer(self.cosmo,z_source=1150),
                                      self.pM)
            elif tracer.quantity == 'cosmic_shear':
                ccl_tracer_dict[tracer.name] = (ccl.WeakLensingTracer(self.cosmo, (tracer.z, tracer.nz)),
                                      self.pM)
            else:
                raise NotImplementedError('Only tracers delta_g, Compton_y, kappa and cosmic_shear supported. Aborting.')

        self.ccl_tracers = ccl_tracer_dict
        
    def getCls (self, tr_i_name, tr_j_name, l_arr):
        """ typ - is a two character string gg, gs,ss, sy, sk etc...
            i,j are indices for g and s"""

        if 'wl' in tr_i_name and 'wl' in tr_j_name or 'wl' in tr_i_name and 'kappa' in tr_j_name or \
                'kappa' in tr_i_name and 'wl' in tr_j_name or 'kappa' in tr_i_name and 'kappa' in tr_j_name:
            if not hasattr(self, 'pk_MMf'):
                Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pM,
                                        normprof1=True,
                                        lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
                self.pk_MMf = Pk
            else:
                Pk = self.pk_MMf
        elif 'wl' in tr_i_name and 'y' in tr_j_name or 'y' in tr_i_name and 'wl' in tr_j_name or \
                'kappa' in tr_i_name and 'y' in tr_j_name or 'y' in tr_i_name and 'kappa' in tr_j_name:
            if not hasattr(self, 'pk_yMf'):
                Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.py, prof2=self.pM,
                                                         normprof1=True, normprof2=True,
                                                         lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
                self.pk_yMf = Pk
            else:
                Pk = self.pk_yMf
        elif 'g' in tr_i_name and 'wl' in tr_j_name or 'wl' in tr_i_name and 'g' in tr_j_name or \
                'kappa' in tr_i_name and 'g' in tr_j_name or 'g' in tr_i_name and 'kappa' in tr_j_name:
            if self.params['HODmod'] == 'zevol':
                if not hasattr(self, 'pk_gMf'):
                    Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.pM,
                                                         normprof1=True, normprof2=True,
                                                         lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
                    self.pk_gMf = Pk
                else:
                    Pk = self.pk_gMf
            else:
                if 'g' in tr_i_name:
                    tr_g_name = tr_i_name
                else:
                    tr_g_name = tr_j_name
                Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_g_name][1], prof2=self.pM,
                                            normprof1=True, normprof2=True,
                                            lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
        elif 'g' in tr_i_name and 'y' in tr_j_name or 'y' in tr_i_name and 'g' in tr_j_name:
            if self.params['HODmod'] == 'zevol':
                if not hasattr(self, 'pk_ygf'):
                    Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.py,
                                                normprof1=True, normprof2=True,
                                                lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
                    self.pk_ygf = Pk
                else:
                    Pk = self.pk_ygf
            else:
                if 'g' in tr_i_name:
                    tr_g_name = tr_i_name
                else:
                    tr_g_name = tr_j_name
                Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_g_name][1], prof2=self.py,
                                       normprof1=True, normprof2=True,
                                       lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
        elif 'g' in tr_i_name and 'g' in tr_j_name:
            if self.params['HODmod'] == 'zevol':
                if not hasattr(self, 'pk_ggf'):
                    Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.pg,
                                           prof_2pt=self.HOD2pt, normprof1=True, normprof2=True,
                                           lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
                    self.pk_ggf = Pk
                else:
                    Pk = self.pk_ggf
            else:
                Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_i_name][1],
                                            prof2=self.ccl_tracers[tr_j_name][1],
                                            prof_2pt=self.HOD2pt, normprof1=True, normprof2=True,
                                            lk_arr=np.log(GSKY_Theory.k_arr), a_arr=GSKY_Theory.a_arr)
        else: ## eg yy
            raise NotImplementedError('Tracer combination {}, {} not implemented. Aborting.'.format(tr_i_name, tr_j_name))

        cls = ccl.angular_cl(self.cosmo, self.ccl_tracers[tr_i_name][0], self.ccl_tracers[tr_j_name][0], l_arr, p_of_k_a=Pk)

        return cls
            
