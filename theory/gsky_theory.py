import numpy as np
import warnings
import pyccl as ccl
import theory.HOD_theory as hod
import theory.SZ_theory as sz
from theory.concentration import ConcentrationDuffy08M500c
from theory.halo_mod_corr import HaloModCorrection

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)
logger.propagate = False

DEFAULT_PARAMS = {
                'corr_halo_mod': True,
                'use_hm_matter': True,
                'corr_halo_mod_cosmo_fid': False,
                'use_EHM': False,
                'EHM_zevol': True,
                'HODmod': 'zevol',
                'mmin': 12.02,
                'mminp': -1.34,
                'm0': 6.6,
                'm0p': -1.43,
                'm1': 13.27,
                'm1p': -0.323,
                'bhydro': 0.2,
                'massdef': 'M200c',
                'pprof': 'Battaglia'
                }

DEFAULT_HMPARAMS_KEYS = ['mmin', 'mminp', 'm0', 'm0p', 'm1', 'm1p', 'bhydro', 'pprof',
                         'massdef', 'corr_halo_mod', 'corr_halo_mod_cosmo_fid', 'HODmod',
                         'use_hm_matter', 'use_EHM', 'EHM_zevol',
                         'zshift_bin0', 'zshift_bin1', 'zshift_bin2', 'zshift_bin3',
                         'zwidth_bin0', 'zwidth_bin1', 'zwidth_bin2', 'zwidth_bin3',
                         'zshift_wl_bin0', 'zshift_wl_bin1', 'zshift_wl_bin2', 'zshift_wl_bin3',
                         'zwidth_wl_bin0', 'zwidth_wl_bin1', 'zwidth_wl_bin2', 'zwidth_wl_bin3',
                         'zshift_g_bin0', 'zshift_g_bin1', 'zshift_g_bin2', 'zshift_g_bin3',
                         'zwidth_g_bin0', 'zwidth_g_bin1', 'zwidth_g_bin2', 'zwidth_g_bin3',
                         'm_bin0', 'm_bin1', 'm_bin2', 'm_bin3',
                         'm_eff',
                         'm_sel_bin0', 'm_sel_bin1', 'm_sel_bin2', 'm_sel_bin3',
                         'm_R_bin0', 'm_R_bin1', 'm_R_bin2', 'm_R_bin3',
                         'A_IA', 'eta_IA', 'z0_IA',            # Intrinsic alignments
                         'cs2', 'R']                        # Effective halo model

class GSKYTheory(object):

    # Wavenumbers and scale factors
    k_arr = np.geomspace(1E-4,1E2,256)
    a_arr = np.linspace(0.2,1,32)

    def __init__ (self, saccfile, params=None, cosmo=None, cosmo_fid=None):
        """ Nz -- list of (zarr,Nzarr) """

        if params is not None:
            self.params = params
            for key, value in DEFAULT_PARAMS.items():
                if key not in params:
                    logger.info('{} not provided.'.format(key))
                    logger.info('Setting {} to default {}.'.format(key, value))
                    self.params[key] = value
        else:
            self.params = DEFAULT_PARAMS

        self.paramnames = DEFAULT_HMPARAMS_KEYS
        if cosmo is None:
            logger.info('No CCL cosmology object provided. Setting up default parameters.')
            logger.info('Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96')
            self.cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)
        else:
            logger.info('CCL cosmology object provided.')
            logger.info('Cosmology = {}.'.format(cosmo))
            self.cosmo = cosmo

        # Setup tracers
        tracer_list = list(saccfile.tracers.values())
        self.tracer_list = tracer_list

        if cosmo_fid is not None:
            logger.info('Fiducial cosmology object provided.')
            logger.info('Cosmology = {}.'.format(cosmo_fid))
            self.cosmo_fid = cosmo_fid
        else:
            self.cosmo_fid = cosmo_fid
            if 'corr_halo_mod_cosmo_fid' in self.params:
                if self.params['corr_halo_mod_cosmo_fid']:
                    logger.info('Requested fiducial cosmolofy for halo model correction but no fiducial cosmology provided. Setting to default.')
                    logger.info('Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96')
                    self.cosmo_fid = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)

        del saccfile

        self._setup_conc()
        self._setup_Cosmo()
        self.check_params()
        self._setup_HM()

    def update_params(self, cosmo, hmparams):

        logger.info('Updating model parameters.')
        self.params.update(hmparams)
        self.set_cosmology(cosmo)
        self.set_HMparams(hmparams)

    def set_HMparams(self, params):

        logger.info('Updating HM parameters.')

        for k in params.keys():
            if k not in self.paramnames:
                raise RuntimeError('Parameter {} not recognized. Aborting.'.format(k))
        self._delete_attrs()
        self.params.update(params)
        self.check_params()
        self._setup_conc()
        self._setup_HM()

    def check_params(self):

        if self.params['pprof'] == 'Battaglia':
            assert self.params['massdef'] == 'M200c', 'Battaglia pressure profile only supported for M200c. Aborting.'
        if self.params['pprof'] == 'Arnaud':
            assert self.params['massdef'] == 'M500c', 'Arnaud pressure profile only supported for M500c. Aborting.'
        
    def set_cosmology(self, cosmo):

        logger.info('Setting cosmology.')

        self._delete_attrs()
        self.cosmo = cosmo
        self._setup_conc()
        self._setup_Cosmo()
        self._setup_HM()
        
    def _setup_Cosmo(self):

        logger.info('Setting up cosmological quantities.')

        # Now we can put together HMCalculator
        # The Tinker 2008 mass function
        self.nM = ccl.halos.MassFuncTinker08(self.cosmo, mass_def=self.hm_def)
        # The Tinker 2010 halo bias
        self.bM = ccl.halos.HaloBiasTinker10(self.cosmo, mass_def=self.hm_def)

        self.hmc = ccl.halos.HMCalculator(self.cosmo, self.nM, self.bM, self.hm_def)

    def _delete_attrs(self):

        for attr in ['pk_MMf', 'pk_yMf', 'pk_gMf', 'pk_ygf', 'pk_ggf', 'rk_hm']:
            if hasattr(self, attr):
                delattr(self, attr)

    def _setup_systematics(self):

        logger.info('Setting up systematics parameters.')

        self.z_c = {}
        for thistracer in self.tracer_list:
            split_name = thistracer.name.split('_')
            if len(split_name) == 2:
                tracer_name = split_name[0]
                tracer_no = split_name[1]
                if tracer_name == 'gc' or tracer_name == 'wl':
                    if 'zwidth_bin{}'.format(tracer_no) in self.params.keys():
                        bin_max = thistracer.nz.max()
                        imax = np.where(thistracer.nz == bin_max)
                        self.z_c['zwidth_bin{}'.format(tracer_no)] = thistracer.z[imax[0][0]]
            else:
                tracer_name = split_name[0]
                if tracer_name == 'gc' or tracer_name == 'wl':
                    if 'zwidth_bin' in self.params.keys():
                        bin_max = thistracer.nz.max()
                        imax = np.where(thistracer.nz == bin_max)
                        self.z_c['zwidth_bin'] = thistracer.z[imax[0][0]]

        if self.z_c == {}:
            logger.info('Nothing to be done.')
        else:
            logger.info('Set up z_c.')

    def _setup_HM(self):

        logger.info('Setting up halo model.')

        self._setup_profiles()
        self._setup_tracers()

    def _setup_conc(self):

        if self.params['massdef'] == 'M200m':
            logger.info('Using M200m.')
            # We will use a mass definition with Delta = 200 times the matter density
            self.hm_def = ccl.halos.MassDef200m()
            # The Duffy 2008 concentration-mass relation
            self.cM = ccl.halos.ConcentrationDuffy08(self.hm_def)
        elif self.params['massdef'] == 'M200c':
            logger.info('Using M200c.')
            # We will use a mass definition with Delta = 200 times the critical density
            self.hm_def = ccl.halos.MassDef200c()
            # The Duffy 2008 concentration-mass relation
            self.cM = ccl.halos.ConcentrationDuffy08(self.hm_def)
        elif self.params['massdef'] == 'M500c':
            logger.info('Using M500c.')
            self.hm_def = ccl.halos.MassDef(500, 'critical')
            self.cM = ConcentrationDuffy08M500c(self.hm_def)
        else:
            raise NotImplementedError('Only mass definitions M200m and M500c supported. Aborting.')

    def _setup_profiles(self):

        logger.info('Setting up halo profiles.')

        self.tracer_quantities = [tr.quantity for tr in self.tracer_list]
        if 'galaxy_shear' in self.tracer_quantities or 'cmb_convergence' in self.tracer_quantities or \
                'cosmic_shear' in self.tracer_quantities or 'kappa' in self.tracer_quantities:
            if 'cosmic_shear' in self.tracer_quantities:
                logger.warning('tracer quantity cosmic_shear will be deprecated soon.')
            if 'kappa' in self.tracer_quantities:
                logger.warning('tracer quantity kappa will be deprecated soon.')
            self.pM = ccl.halos.profiles.HaloProfileNFW(self.cM)
        if 'cmb_tSZ' in self.tracer_quantities or 'Compton_y' in self.tracer_quantities:
            if 'Compton_y' in self.tracer_quantities:
                logger.warning('tracer quantity Compton_y will be deprecated soon.')
            if self.params['pprof'] == 'Arnaud':
                logger.info('Using Arnaud profile.')
                self.py = sz.HaloProfileArnaud(b_hydro=self.params['bhydro'])
            elif self.params['pprof'] == 'Battaglia':
                logger.info('Using Battaglia profile.')
                self.py = sz.HaloProfileBattaglia()
            else:
                self.py = None
                warnings.warn('Only pressure profiles Arnaud and Battaglia implemented.', RuntimeWarning)
        if 'galaxy_density' in self.tracer_quantities or 'delta_g' in self.tracer_quantities:
            if 'delta_g' in self.tracer_quantities:
                logger.warning('tracer quantity delta_g will be deprecated soon.')
            self.HOD2pt = hod.Profile2ptHOD()
            if self.params['HODmod'] == 'zevol':
                self.pg = hod.HaloProfileHOD(c_M_relation=self.cM,
                                        lMmin=self.params['mmin'], lMminp=self.params['mminp'],
                                        lM0=self.params['m0'], lM0p=self.params['m0p'],
                                        lM1=self.params['m1'], lM1p=self.params['m1p'])
        
    def _setup_tracers(self):

        logger.info('Setting up tracers.')

        p = self.params

        self._setup_systematics()

        ccl_tracer_dict = {}

        for i, tracer in enumerate(self.tracer_list):
            if tracer.quantity == 'galaxy_density' or tracer.quantity == 'delta_g':
                if tracer.quantity == 'delta_g':
                    logger.warning('tracer quantity {} will be deprecated soon.'.format(tracer.quantity))
                split_name = tracer.name.split('_')
                if len(split_name) == 2:
                    tracer_no = split_name[1]
                    # Bias
                    if 'bb_{}'.format(tracer_no) in p.keys():
                        logger.info('Galaxy bias array provided for {}.'.format(tracer.name))
                        bias_tup = (p['bz_{}'.format(tracer_no)], p['bb_{}'.format(tracer_no)])
                    else:
                        logger.info('Galaxy bias array not provided for {}. Setting to unity.'.format(tracer.name))
                        bias_tup = (tracer.z, np.ones_like(tracer.z))
                    # z_shift parameter
                    if ('zshift_bin{}'.format(tracer_no) in p.keys()) and ('zwidth_bin{}'.format(tracer_no) in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_bin{}'.format(tracer_no)]) * (
                                1 + p['zwidth_bin{}'.format(tracer_no)]) + \
                                p['zshift_bin{}'.format(tracer_no)] + self.z_c['zwidth_bin{}'.format(tracer_no)]

                    elif ('zshift_g_bin{}'.format(tracer_no) in p.keys()) and ('zwidth_g_bin{}'.format(tracer_no) in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_g_bin{}'.format(tracer_no)]) * (
                                1 + p['zwidth_g_bin{}'.format(tracer_no)]) + \
                                p['zshift_g_bin{}'.format(tracer_no)] + self.z_c['zwidth_g_bin{}'.format(tracer_no)]

                    elif 'zshift_bin{}'.format(tracer_no) in p.keys():
                        zbins = tracer.z + p['zshift_bin{}'.format(tracer_no)]

                    elif 'zshift_g_bin{}'.format(tracer_no) in p.keys():
                        zbins = tracer.z + p['zshift_g_bin{}'.format(tracer_no)]

                    else:
                        zbins = tracer.z

                else:
                    # Bias
                    if 'bb' in p.keys():
                        logger.info('Galaxy bias array provided for {}.'.format(tracer.name))
                        bias_tup = (p['bz'], p['bb'])
                    else:
                        logger.info('Galaxy bias array not provided for {}. Setting to unity.'.format(tracer.name))
                        bias_tup = (tracer.z, np.ones_like(tracer.z))
                    # z_shift parameter
                    if ('zshift_bin' in p.keys()) and ('zwidth_bin' in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_bin']) * (1 + p['zwidth_bin']) + \
                                p['zshift_bin'] + self.z_c['zwidth_bin']

                    elif ('zshift_g_bin' in p.keys()) and ('zwidth_g_bin' in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_g_bin']) * (1 + p['zwidth_bin']) + \
                                p['zshift_g_bin'] + self.z_c['zwidth_g_bin']

                    elif 'zshift_bin' in p.keys():
                        zbins = tracer.z + p['zshift_bin']

                    elif 'zshift_g_bin' in p.keys():
                        zbins = tracer.z + p['zshift_g_bin']

                    else:
                        zbins = tracer.z
                # pz method
                if 'pzMethod' in p.keys():
                    if p['pzMethod'] != 'COSMOS30':
                        nz = tracer.extra_columns[p['pzMethod']]
                    else:
                        nz = tracer.nz
                else:
                    nz = tracer.nz

                if p['HODmod'] == 'zevol':
                    ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.NumberCountsTracer(self.cosmo, False,
                                                        (zbins[zbins>=0.], nz[zbins>=0.]),
                                                        bias=bias_tup),
                                                    'prof': self.pg}
                else:
                    ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.NumberCountsTracer(self.cosmo, False,
                                                                           (zbins[zbins>=0.], nz[zbins>=0.]),
                                                                           bias=bias_tup),
                                                    'prof': hod.HaloProfileHOD(c_M_relation=self.cM,
                                                                       lMmin=p['mmin'], lMminp=p['mminp'],
                                                                       lM0=p['m0'], lM0p=p['m0p'],
                                                                       lM1=p['m1'], lM1p=p['m1p'])}
            elif tracer.quantity == 'cmb_tSZ' or tracer.quantity == 'Compton_y':
                if tracer.quantity == 'Compton_y':
                    logger.warning('tracer quantity {} will be deprecated soon.'.format(tracer.quantity))
                ccl_tracer_dict[tracer.name] = {'ccl_tracer': sz.SZTracer(self.cosmo),
                                                'prof': self.py}
            elif tracer.quantity == 'cmb_convergence' or tracer.quantity == 'kappa':
                if tracer.quantity == 'kappa':
                    logger.warning('tracer quantity {} will be deprecated soon.'.format(tracer.quantity))
                ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.CMBLensingTracer(self.cosmo,z_source=1150),
                                                'prof': self.pM}
            elif tracer.quantity == 'galaxy_shear' or tracer.quantity == 'cosmic_shear':
                if tracer.quantity == 'cosmic_shear':
                    logger.warning('tracer quantity {} will be deprecated soon.'.format(tracer.quantity))

                split_name = tracer.name.split('_')
                if len(split_name) == 2:
                    tracer_no = split_name[1]
                    # z_shift parameter
                    if ('zshift_bin{}'.format(tracer_no) in p.keys()) and ('zwidth_bin{}'.format(tracer_no) in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_bin{}'.format(tracer_no)]) * (
                                1 + p['zwidth_bin{}'.format(tracer_no)]) + \
                                p['zshift_bin{}'.format(tracer_no)] + self.z_c['zwidth_bin{}'.format(tracer_no)]

                    elif ('zshift_wl_bin{}'.format(tracer_no) in p.keys()) and ('zwidth_wl_bin{}'.format(tracer_no) in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_wl_bin{}'.format(tracer_no)]) * (
                                1 + p['zwidth_wl_bin{}'.format(tracer_no)]) + \
                                p['zshift_wl_bin{}'.format(tracer_no)] + self.z_c['zwidth_wl_bin{}'.format(tracer_no)]

                    elif 'zshift_bin{}'.format(tracer_no) in p.keys():
                        zbins = tracer.z + p['zshift_bin{}'.format(tracer_no)]

                    elif 'zshift_wl_bin{}'.format(tracer_no) in p.keys():
                        zbins = tracer.z + p['zshift_wl_bin{}'.format(tracer_no)]

                    else:
                        zbins = tracer.z

                else:
                    # z_shift parameter
                    if ('zshift_bin' in p.keys()) and ('zwidth_bin' in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_bin']) * (1 + p['zwidth_bin']) + \
                                p['zshift_bin'] + self.z_c['zwidth_bin']

                    elif ('zshift_wl_bin' in p.keys()) and ('zwidth_wl_bin' in p.keys()):
                        zbins = (tracer.z - self.z_c['zwidth_wl_bin']) * (1 + p['zwidth_wl_bin']) + \
                                p['zshift_wl_bin'] + self.z_c['zwidth_wl_bin']

                    elif 'zshift_bin' in p.keys():
                        zbins = tracer.z + p['zshift_bin']

                    elif 'zshift_wl_bin' in p.keys():
                        zbins = tracer.z + p['zshift_wl_bin']

                    else:
                        zbins = tracer.z

                # pz method
                if 'pzMethod' in p.keys():
                    if p['pzMethod'] != 'COSMOS30':
                        nz = tracer.extra_columns[p['pzMethod']]
                    else:
                        nz = tracer.nz
                else:
                    nz = tracer.nz

                # Intrinsic alignments
                if 'A_IA' in p.keys():
                    if 'eta_IA' in p.keys():
                        assert 'z0_IA' in p.keys(), 'Redshift-dependent intrinsic alignments requested but z0_IA not ' \
                                                   'supplied. Aborting.'
                        A_IA_z = p['A_IA']*((1. + zbins[zbins>=0.])/(1. + p['z0_IA']))**p['eta_IA']
                        ia_bias = (zbins[zbins>=0.], A_IA_z)
                    else:
                        A_IA_z = p['A_IA']*np.ones_like(zbins[zbins>=0.])
                        ia_bias = (zbins[zbins>=0.], A_IA_z)
                else:
                    ia_bias = None

                if 'm_bin{}'.format(tracer_no) in p.keys():
                    if 'm_sel_bin{}'.format(tracer_no) not in p.keys():
                        ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.WeakLensingTracer(self.cosmo,
                                                            (zbins[zbins>=0.], nz[zbins>=0.]), ia_bias=ia_bias),
                                                        'prof': self.pM,
                                                        'm': p['m_bin{}'.format(tracer_no)]}
                    else:
                        ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.WeakLensingTracer(self.cosmo,
                                                                                            (zbins[zbins >= 0.],
                                                                                             nz[zbins >= 0.]),
                                                                                            ia_bias=ia_bias),
                                                        'prof': self.pM,
                                                        'm': p['m_bin{}'.format(tracer_no)],
                                                        'm_sel': p['m_sel_bin{}'.format(tracer_no)],
                                                        'm_R': p['m_R_bin{}'.format(tracer_no)]}
                elif 'm_eff' in p.keys():
                    if 'm_sel_bin{}'.format(tracer_no) not in p.keys():
                        ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.WeakLensingTracer(self.cosmo,
                                                                                            (zbins[zbins >= 0.],
                                                                                             nz[zbins >= 0.]),
                                                                                            ia_bias=ia_bias),
                                                        'prof': self.pM,
                                                        'm': p['m_eff']}
                    else:
                        ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.WeakLensingTracer(self.cosmo,
                                                                                            (zbins[zbins >= 0.],
                                                                                             nz[zbins >= 0.]),
                                                                                            ia_bias=ia_bias),
                                                        'prof': self.pM,
                                                        'm': p['m_eff'],
                                                        'm_sel': p['m_sel_bin{}'.format(tracer_no)],
                                                        'm_R': p['m_R_bin{}'.format(tracer_no)]}
                else:
                    if 'm_sel_bin{}'.format(tracer_no) not in p.keys():
                        ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.WeakLensingTracer(self.cosmo,
                                                            (zbins[zbins >= 0.], nz[zbins >= 0.]), ia_bias=ia_bias),
                                                        'prof': self.pM}
                    else:
                        ccl_tracer_dict[tracer.name] = {'ccl_tracer': ccl.WeakLensingTracer(self.cosmo,
                                                                                            (zbins[zbins >= 0.],
                                                                                             nz[zbins >= 0.]),
                                                                                            ia_bias=ia_bias),
                                                        'prof': self.pM,
                                                        'm_sel': p['m_sel_bin{}'.format(tracer_no)],
                                                        'm_R': p['m_R_bin{}'.format(tracer_no)]}
            else:
                raise NotImplementedError('Only tracers galaxy_density, cmb_tSZ, cmb_convergence and galaxy_shear supported. Aborting.')

        self.ccl_tracers = ccl_tracer_dict
        
    def getCls (self, tr_i_name, tr_j_name, l_arr):
        """ typ - is a two character string gg, gs,ss, sy, sk etc...
            i,j are indices for g and s"""

        if self.params['corr_halo_mod']:
            logger.info('Correcting halo model Pk with HALOFIT ratio.')
            if not hasattr(self, 'rk_hm'):
                if not hasattr(self, 'pM'):
                    logger.info('Computing matter density profile.')
                    self.pM = ccl.halos.profiles.HaloProfileNFW(self.cM)
                logger.info('Computing halo model correction.')
                if 'corr_halo_mod_cosmo_fid' in self.params:
                    if self.params['corr_halo_mod_cosmo_fid']:
                        logger.info('Using fiducial cosmology for halo model correction.')
                        # Now we can put together HMCalculator
                        # The Tinker 2008 mass function
                        nM_fid = ccl.halos.MassFuncTinker08(self.cosmo_fid, mass_def=self.hm_def)
                        # The Tinker 2010 halo bias
                        bM_fid = ccl.halos.HaloBiasTinker10(self.cosmo_fid, mass_def=self.hm_def)
                        hmc_fid = ccl.halos.HMCalculator(self.cosmo_fid, nM_fid, bM_fid, self.hm_def)
                        HMCorr = HaloModCorrection(self.cosmo_fid, hmc_fid, self.pM, k_range=[1e-4, 1e2], nlk=256,
                                                   z_range=[0., 3.], nz=50)
                    else:
                        logger.info('Using current cosmology for halo model correction.')
                        HMCorr = HaloModCorrection(self.cosmo, self.hmc, self.pM, k_range=[1e-4, 1e2], nlk=256,
                                                   z_range=[0., 3.], nz=50)
                else:
                    logger.info('Using current cosmology for halo model correction.')
                    HMCorr = HaloModCorrection(self.cosmo, self.hmc, self.pM, k_range=[1e-4, 1e2], nlk=256,
                                               z_range=[0., 3.], nz=50)
                self.rk_hm = HMCorr.rk_interp(GSKYTheory.k_arr, GSKYTheory.a_arr)[::-1]

        if 'wl' in tr_i_name and 'wl' in tr_j_name or 'wl' in tr_i_name and 'kappa' in tr_j_name or \
                'kappa' in tr_i_name and 'wl' in tr_j_name or 'kappa' in tr_i_name and 'kappa' in tr_j_name:
            if not hasattr(self, 'pk_MMf'):
                if self.params['use_hm_matter']:
                    logger.info('Using halomodel for matter power spectra.')
                    if self.params['use_EHM']:
                        ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                      z_evol=self.params['EHM_zevol'])
                        pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pM, normprof1=True,
                                                              lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                              p_of_k_a=pk_eff_2d)
                    else:
                        if not self.params['corr_halo_mod']:
                            Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pM, normprof1=True,
                                                    lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                        else:
                            Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                                                      self.pM, normprof1=True)
                            Pk_arr *= self.rk_hm
                            Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                          cosmo=self.cosmo, is_logp=False)
                    self.pk_MMf = Pk
                else:
                    logger.info('Using halofit for matter power spectra.')
                    Pk_arr = np.array([ccl.nonlin_matter_power(self.cosmo, GSKYTheory.k_arr, a) for a in GSKYTheory.a_arr])
                    Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                  cosmo=self.cosmo, is_logp=False)
                    self.pk_MMf = Pk
            else:
                Pk = self.pk_MMf
        elif 'wl' in tr_i_name and 'y' in tr_j_name or 'y' in tr_i_name and 'wl' in tr_j_name or \
                'kappa' in tr_i_name and 'y' in tr_j_name or 'y' in tr_i_name and 'kappa' in tr_j_name:
            if not hasattr(self, 'pk_yMf'):
                if self.params['use_EHM']:
                    ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                  z_evol=self.params['EHM_zevol'])
                    pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                    Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.py, prof2=self.pM,
                                                          normprof1=False, normprof2=True,
                                                          lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                          p_of_k_a=pk_eff_2d)
                else:
                    if not self.params['corr_halo_mod']:
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.py, prof2=self.pM,
                                                                 normprof1=False, normprof2=True,
                                                                 lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                    else:
                        Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                                                  self.py, prof2=self.pM, normprof1=False, normprof2=True)
                        Pk_arr *= self.rk_hm
                        Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                      cosmo=self.cosmo, is_logp=False)
                self.pk_yMf = Pk

            else:
                Pk = self.pk_yMf
        elif 'g' in tr_i_name and 'wl' in tr_j_name or 'wl' in tr_i_name and 'g' in tr_j_name or \
                'kappa' in tr_i_name and 'g' in tr_j_name or 'g' in tr_i_name and 'kappa' in tr_j_name:
            if self.params['HODmod'] == 'zevol':
                if not hasattr(self, 'pk_gMf'):
                    if self.params['use_EHM']:
                        ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                      z_evol=self.params['EHM_zevol'])
                        pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.pM,
                                                              normprof1=True, normprof2=True,
                                                              lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                              p_of_k_a=pk_eff_2d)
                    else:
                        if not self.params['corr_halo_mod']:
                            Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.pM, normprof1=True,
                                                        normprof2=True, lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                        else:
                            Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                                                      self.pg, prof2=self.pM, normprof1=True, normprof2=True)
                            Pk_arr *= self.rk_hm
                            Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                          cosmo=self.cosmo, is_logp=False)
                    self.pk_gMf = Pk

                else:
                    Pk = self.pk_gMf
            else:
                if 'g' in tr_i_name:
                    tr_g_name = tr_i_name
                else:
                    tr_g_name = tr_j_name
                if self.params['use_EHM']:
                    ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                  z_evol=self.params['EHM_zevol'])
                    pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                    Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_g_name]['prof'],
                                                          prof2=self.pM, normprof1=True, normprof2=True,
                                                          lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                          p_of_k_a=pk_eff_2d)
                else:
                    if not self.params['corr_halo_mod']:
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_g_name]['prof'],
                                                    prof2=self.pM, normprof1=True, normprof2=True,
                                                    lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                    else:
                        Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                                                  self.ccl_tracers[tr_g_name]['prof'], prof2=self.pM,
                                                                  normprof1=True, normprof2=True)
                        Pk_arr *= self.rk_hm
                        Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                      cosmo=self.cosmo, is_logp=False)
        elif 'g' in tr_i_name and 'y' in tr_j_name or 'y' in tr_i_name and 'g' in tr_j_name:
            if self.params['HODmod'] == 'zevol':
                if not hasattr(self, 'pk_ygf'):
                    if self.params['use_EHM']:
                        ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                      z_evol=self.params['EHM_zevol'])
                        pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.py,
                                                              normprof1=True, normprof2=False,
                                                              lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                              p_of_k_a=pk_eff_2d)
                    else:
                        if not self.params['corr_halo_mod']:
                            Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.py, normprof1=True,
                                                        normprof2=False, lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                        else:
                            Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                                                      self.pg, prof2=self.py, normprof1=True, normprof2=False)
                            Pk_arr *= self.rk_hm
                            Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                          cosmo=self.cosmo, is_logp=False)
                    self.pk_ygf = Pk

                else:
                    Pk = self.pk_ygf
            else:
                if 'g' in tr_i_name:
                    tr_g_name = tr_i_name
                else:
                    tr_g_name = tr_j_name
                Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_g_name]['prof'], prof2=self.py,
                                       normprof1=True, normprof2=False,
                                       lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
        elif 'g' in tr_i_name and 'g' in tr_j_name:
            if self.params['HODmod'] == 'zevol':
                if not hasattr(self, 'pk_ggf'):
                    if self.params['use_EHM']:
                        ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                  z_evol=self.params['EHM_zevol'])
                        pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.pg,
                                                              prof_2pt=self.HOD2pt, normprof1=True, normprof2=True,
                                                              lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                              p_of_k_a=pk_eff_2d)
                    else:
                        if not self.params['corr_halo_mod']:
                            Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.pg, prof2=self.pg,
                                                   prof_2pt=self.HOD2pt, normprof1=True, normprof2=True,
                                                   lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                        else:
                            Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                        self.pg, prof_2pt=self.HOD2pt, prof2=self.pg, normprof1=True, normprof2=True)
                            Pk_arr *= self.rk_hm
                            Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                        cosmo=self.cosmo, is_logp=False)

                    self.pk_ggf = Pk
                else:
                    Pk = self.pk_ggf
            else:
                if self.params['use_EHM']:
                    ptc = ccl.halos.PTNLEffCalculator(self.cosmo, self.params['cs2'], self.params['R'],
                                                  z_evol=self.params['EHM_zevol'])
                    pk_eff_2d = ccl.halos.get_pt_eff_pk2d(self.cosmo, ptc)
                    Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_i_name]['prof'],
                                                          prof2=self.ccl_tracers[tr_j_name]['prof'],
                                                          prof_2pt=self.HOD2pt, normprof1=True, normprof2=True,
                                                          lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr,
                                                          p_of_k_a=pk_eff_2d)
                else:
                    if not self.params['corr_halo_mod']:
                        Pk = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, self.ccl_tracers[tr_i_name]['prof'],
                                                    prof2=self.ccl_tracers[tr_j_name]['prof'],
                                                    prof_2pt=self.HOD2pt, normprof1=True, normprof2=True,
                                                    lk_arr=np.log(GSKYTheory.k_arr), a_arr=GSKYTheory.a_arr)
                    else:
                        Pk_arr = ccl.halos.halomod_power_spectrum(self.cosmo, self.hmc, GSKYTheory.k_arr, GSKYTheory.a_arr,
                                                                  self.ccl_tracers[tr_i_name]['prof'], prof_2pt=self.HOD2pt,
                                                                  prof2=self.ccl_tracers[tr_j_name]['prof'],
                                                                  normprof1=True, normprof2=True)
                        Pk_arr *= self.rk_hm
                        Pk = ccl.Pk2D(a_arr=GSKYTheory.a_arr, lk_arr=np.log(GSKYTheory.k_arr), pk_arr=Pk_arr,
                                      cosmo=self.cosmo, is_logp=False)
        else: ## eg yy
            logger.warning('Tracer combination {}, {} not implemented. Returning zero.'.format(tr_i_name, tr_j_name))
            return np.zeros_like(l_arr)

        cls = ccl.angular_cl(self.cosmo, self.ccl_tracers[tr_i_name]['ccl_tracer'], self.ccl_tracers[tr_j_name]['ccl_tracer'],
                             l_arr, p_of_k_a=Pk)

        if 'wl' in tr_i_name and 'm' in self.ccl_tracers[tr_i_name].keys():
            cls *= (1. + self.ccl_tracers[tr_i_name]['m'])
        if 'wl' in tr_j_name and 'm' in self.ccl_tracers[tr_j_name].keys():
            cls *= (1. + self.ccl_tracers[tr_j_name]['m'])

        if 'wl' in tr_i_name and 'm_sel' in self.ccl_tracers[tr_i_name].keys():
            cls *= (1. + self.ccl_tracers[tr_i_name]['m_sel'] + self.ccl_tracers[tr_i_name]['m_R'])
        if 'wl' in tr_j_name and 'm_sel' in self.ccl_tracers[tr_j_name].keys():
            cls *= (1. + self.ccl_tracers[tr_j_name]['m_sel'] + self.ccl_tracers[tr_j_name]['m_R'])

        return cls
            
