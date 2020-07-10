import numpy as np
from theory.gsky_theory import GSKYTheory
import sacc
import pyccl as ccl
import theory.theory_util as tutil
from theory.theory_util import ClInterpolator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_params(params, paramtype):
    params_subset = {}

    if paramtype == 'cosmo':
        KEYS = ['Omega_c', 'Omega_b', 'h', 'n_s', 'sigma8', 'A_s', 'Omega_k', 'Omega_g', 'Neff', 'm_nu',
                'mnu_type', 'w0', 'wa', 'bcm_log10Mc', 'bcm_etab', 'bcm_ks', 'z_mg', 'df_mg',
                'transfer_function', 'matter_power_spectrum', 'baryons_power_spectrum', 'mass_function',
                'halo_concentration', 'emulator_neutrinos']
    elif paramtype == 'hmparams':
        KEYS = ['HODmod', 'massdef', 'pprof', 'corr_halo_mod', 'corr_halo_mod_cosmo_fid',
                'use_hm_matter', 'use_EHM', 'EHM_zevol',
                'mmin', 'mminp', 'm0', 'm0p', 'm1', 'm1p', 'bhydro',
                'zshift_bin0', 'zshift_bin1', 'zshift_bin2', 'zshift_bin3',
                'zwidth_bin0', 'zwidth_bin1', 'zwidth_bin2', 'zwidth_bin3',
                'm_bin0', 'm_bin1', 'm_bin2', 'm_bin3',
                'A_IA', 'eta', 'z0_IA',  # Intrinsic alignments
                'cs2', 'R']
    else:
        return

    for key in KEYS:
        if key in params:
            params_subset[key] = params[key]

    return params_subset

class GSKYCore(object):

    def __init__ (self, saccfile, config):

        logger.info('Setting up GSKYCore.')
        if not type(saccfile) == sacc.sacc.Sacc:
            saccfile = sacc.Sacc.load_fits(saccfile)
        self.saccfile = saccfile
        self.ells = config['ells']
        self.mapping = config['param_mapping']
        if config['constants'] is not None:
            logger.info('Constants {} provided.'.format(config['constants']))
            self.constants = config['constants']
        else:
            self.constants = {}
        if config['cosmo'] is not None:
            logger.info('Fiducial cosmology with parameters {} provided.'.format(config['cosmo']))
            self.fid_cosmo = config['cosmo']
        else:
            self.fid_cosmo = None
        if config['hmparams'] is not None:
            logger.info('Fiducial HM parameters {} provided.'.format(config['hmparams']))
            self.fid_HMparams = config['hmparams']
        else:
            self.fid_HMparams = {}
        if 'conv_win' in config.keys():
            self.conv_win = config['conv_win']
        else:
            self.conv_win = None
        self.trc_combs = config['trc_combs']

        # self.gskytheor = GSKYTheory(self.saccfile, self.fid_HMparams, self.fid_cosmo)

    def computeTheory(self, p):

        params = self.constants.copy()
        for k, v in self.mapping.items():
            params[k] = p[v]

        cosmo_params = get_params(params, 'cosmo')
        HMparams = get_params(params, 'hmparams')

        if (cosmo_params.keys() & self.mapping.keys()) != set([]):
            cosmo = ccl.Cosmology(**cosmo_params)
        else:
            cosmo = self.fid_cosmo
        if (HMparams.keys() & self.mapping.keys()) == set([]):
            HMparams = self.fid_HMparams

        if 'corr_halo_mod_cosmo_fid' in HMparams:
            if HMparams['corr_halo_mod_cosmo_fid']:
                gskytheor = GSKYTheory(self.saccfile, HMparams, cosmo, self.fid_cosmo)
            else:
                gskytheor = GSKYTheory(self.saccfile, HMparams, cosmo)
        else:
            gskytheor = GSKYTheory(self.saccfile, HMparams, cosmo)

        cls = np.zeros_like(self.saccfile.mean)

        for tr_i, tr_j in self.trc_combs:
            logger.info('Computing theory prediction for tracers {}, {}.'.format(tr_i, tr_j))

            if 'wl' not in tr_i and 'wl' not in tr_j:
                datatype = 'cl_00'
            elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
                datatype = 'cl_0e'
            else:
                datatype = 'cl_ee'

            indx_curr = self.saccfile.indices(data_type=datatype, tracers=(tr_i, tr_j))
            if indx_curr != np.array([]):
                if self.ells != 'NONE':
                    if self.conv_win:
                        # Get window
                        win_curr = self.saccfile.get_bandpower_windows(indx_curr)
                        ell_max = int(np.ceil(np.amax(win_curr.values)))
                        itp = ClInterpolator(self.ells, ell_max)
                        cl_temp = gskytheor.getCls(tr_i, tr_j, itp.ls_eval)
                    else:
                        cl_temp = gskytheor.getCls(tr_i, tr_j, self.ells)
                else:
                    ells_curr, _ = self.saccfile.get_ell_cl(datatype, tr_i, tr_j, return_cov=False)
                    if self.conv_win:
                        # Get window
                        win_curr = self.saccfile.get_bandpower_windows(indx_curr)
                        ell_max = np.amax(win_curr.values)
                        itp = ClInterpolator(ells_curr, ell_max)
                        cl_temp = gskytheor.getCls(tr_i, tr_j, itp.ls_eval)
                    else:
                        cl_temp = gskytheor.getCls(tr_i, tr_j, ells_curr)

                if self.conv_win:
                    cl_temp = tutil.interp_and_convolve(cl_temp, win_curr, itp)

                cls[indx_curr] = cl_temp

            else:
                logger.warning('Empty tracer combination. Check tracer order.')

        del gskytheor

        return cls

    def setup(self):

        pass
