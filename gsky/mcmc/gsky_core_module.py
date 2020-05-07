import numpy as np
from theory.gsky_theory import GSKYTheory
import sacc
import pyccl as ccl
from cosmoHammer.exceptions import LikelihoodComputationException

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSKYCore(object):

    def __init__ (self, saccfile, ells, mapping, constants=None, HMparams=None, cosmo=None):

        logger.info('Setting up GSKYCore.')
        if not type(saccfile) == sacc.sacc.Sacc:
            saccfile = sacc.Sacc.load_fits(saccfile)
        self.saccfile = saccfile
        self.ells = ells
        self.mapping = mapping
        if constants is not None:
            logger.info('Constants {} provided.'.format(constants))
            self.constants = constants
        else:
            self.constants = {}
        if cosmo is not None:
            logger.info('Fiducial cosmology with parameters {} provided.'.format(cosmo))
            self.fid_cosmo = cosmo
        else:
            self.fid_cosmo = None
        if HMparams is not None:
            logger.info('Fiducial HM parameters {} provided.'.format(HMparams))
            self.fid_HMparams = HMparams
        else:
            self.fid_HMparams = {}

        self.gskytheor = GSKYTheory(self.saccfile, self.fid_HMparams, self.fid_cosmo)

    def __call__(self, ctx):

        # Get the parameters from the context
        p = ctx.getParams()

        params = self.constants.copy()
        for k, v in self.mapping.items():
            params[k] = p[v]

        cosmo_params = self.get_params(params, 'cosmo')
        HMparams = self.get_params(params, 'HMparams')

        try:
            if (cosmo_params.keys() & self.mapping.keys()) != set([]):
                cosmo = ccl.Cosmology(**cosmo_params)
            else:
                cosmo = self.fid_cosmo
            if (HMparams.keys() & self.mapping.keys()) == set([]):
                HMparams = self.fid_HMparams

            self.gskytheor.update_params(cosmo, HMparams)

            cls = np.zeros_like(self.saccfile.mean)

            for tr_i, tr_j in self.saccfile.get_tracer_combinations():
                logger.info('Computing theory prediction for tracers {}, {}.'.format(tr_i, tr_j))
                cl_temp = self.gskytheor.getCls(tr_i, tr_j, self.ells)
                if 'wl' not in tr_i and 'wl' not in tr_j:
                    indx = self.saccfile.indices('cl_00', (tr_i, tr_j))
                elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
                    indx = self.saccfile.indices('cl_0e', (tr_i, tr_j))
                else:
                    indx = self.saccfile.indices('cl_ee', (tr_i, tr_j))

                cls[indx] = cl_temp

            # Add the theoretical cls to the context
            ctx.add('obs_theory', cls)

        except BaseException as e:
            logger.error('{} for parameter set {}.'.format(e, p))
            raise LikelihoodComputationException()

    def get_params(self, params, paramtype):

        params_subset = {}

        if paramtype == 'cosmo':
            KEYS = ['Omega_c', 'Omega_b', 'h', 'n_s', 'sigma8', 'A_s', 'Omega_k', 'Omega_g', 'Neff', 'm_nu',
                                'mnu_type', 'w0', 'wa', 'bcm_log10Mc', 'bcm_etab', 'bcm_ks', 'z_mg', 'df_mg',
                                'transfer_function', 'matter_power_spectrum', 'baryons_power_spectrum',
                                'mass_function', 'halo_concentration', 'emulator_neutrinos']
        elif paramtype == 'HMparams':
            KEYS = ['lmmin_0', 'lmmin_1', 'sigm_0', 'sigm_1', 'm0_0', 'm0_1', 'm1_0', 'm1_1',
                  'alpha_0', 'alpha_1', 'fc_0', 'fc_1', 'lmmin', 'lmminp', 'm1', 'm1p', 'm0',
                  'm0p', 'zfid']
        else:
            return

        for key in KEYS:
            if key in params:
                params_subset[key] = params[key]

        return params_subset

    def setup(self):

        pass
