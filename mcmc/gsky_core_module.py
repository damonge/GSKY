import numpy as np
from theory.gsky_theory import GSKYTheory
import sacc
import pyccl as ccl
from cosmoHammer.exceptions import LikelihoodComputationException
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
        KEYS = ['HODmod', 'massdef', 'pprof', 'corr_halo_mod',
                'mmin', 'mminp', 'm0', 'm0p', 'm1', 'm1p', 'bhydro',
                'zshift_bin0', 'zshift_bin1', 'zshift_bin2', 'zshift_bin3',
                'zwidth_bin0', 'zwidth_bin1', 'zwidth_bin2', 'zwidth_bin3']
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

    def __call__(self, ctx):

        # Get the parameters from the context
        p = ctx.getParams()

        params = self.constants.copy()
        for k, v in self.mapping.items():
            params[k] = p[v]

        cosmo_params = get_params(params, 'cosmo')
        HMparams = get_params(params, 'hmparams')

        try:
            if (cosmo_params.keys() & self.mapping.keys()) != set([]):
                cosmo = ccl.Cosmology(**cosmo_params)
            else:
                cosmo = self.fid_cosmo
            if (HMparams.keys() & self.mapping.keys()) == set([]):
                HMparams = self.fid_HMparams

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

                if self.ells != 'NONE':
                    if self.conv_win:
                        # Get window
                        win = self.saccfile.get_tag('window', tracers=(tr_i, tr_j), data_type=datatype)
                        if type(win) is list:
                            win = win[0]
                        ell_max = win.values.shape[0]
                        itp = ClInterpolator(self.ells, np.amax(ell_max))
                        cl_temp = gskytheor.getCls(tr_i, tr_j, itp.ls_eval)
                    else:
                        cl_temp = gskytheor.getCls(tr_i, tr_j, self.ells)
                else:
                    ells_curr = np.array(self.saccfile.get_tag('ell', tracers=(tr_i, tr_j), data_type=datatype))
                    if self.conv_win:
                        # Get window
                        win = self.saccfile.get_tag('window', tracers=(tr_i, tr_j), data_type=datatype)
                        if type(win) is list:
                            win = win[0]
                        ell_max = win.values.shape[0]
                        itp = ClInterpolator(self.ells, np.amax(ell_max))
                        cl_temp = gskytheor.getCls(tr_i, tr_j, itp.ls_eval)
                    else:
                        cl_temp = gskytheor.getCls(tr_i, tr_j, ells_curr)

                indx = self.saccfile.indices(datatype, (tr_i, tr_j))

                if self.conv_win:
                    cl_temp = tutil.convolve(cl_temp, win, itp)

                cls[indx] = cl_temp

            # Add the theoretical cls to the context
            ctx.add('obs_theory', cls)

        except BaseException as e:
            logger.error('{} for parameter set {}.'.format(e, p))
            raise LikelihoodComputationException()

    def setup(self):

        pass
