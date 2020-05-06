import numpy as np
from .gsky_theory import GSKYTheory
import sacc
import pyccl as ccl

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSKYPrediction(object):

    def __init__ (self, saccfile, ells='NONE', hmparams=None, cosmo=None):

        self.setup(saccfile, ells, hmparams, cosmo)

    def get_prediction(self, params, trc_combs=None, datatype=None):

        if trc_combs is None:
            logger.info('Computing theory predictions for all tracer combinations in sacc.')
            trc_combs = self.saccfile.get_tracer_combinations()
        else:
            logger.info('Computing theory predictions for tracer combinations {}.'.format(trc_combs))

        if 'cosmo' in params.keys():
            cosmo_params = params['cosmo']
        else:
            cosmo_params = {}
        if 'hmparams' in params.keys():
            hmparams = params['hmparams']
        else:
            hmparams = {}

        if cosmo_params != {} and hmparams != {}:
            cosmo = ccl.Cosmology(**cosmo_params)
            self.gskytheor.update_params(cosmo, hmparams)
        elif cosmo_params == {} and hmparams != {}:
            self.gskytheor.set_HMparams(hmparams)
        elif cosmo_params != {} and hmparams == {}:
            cosmo = ccl.Cosmology(**cosmo_params)
            self.gskytheor.set_cosmology(cosmo)
        else:
            raise RuntimeError('Either hmparams or cosmo_params need to be provided. Aborting.')

        cls = np.zeros_like(self.saccfile.mean)

        for tr_i, tr_j in trc_combs:
            logger.info('Computing theory prediction for tracers {}, {}.'.format(tr_i, tr_j))
            if self.ells != 'NONE':
                cl_temp = self.gskytheor.getCls(tr_i, tr_j, self.ells)
            else:
                ells_curr = np.array(self.saccfile.get_tag('ell', tracers=(tr_i, tr_j), data_type=datatype))
                cl_temp = self.gskytheor.getCls(tr_i, tr_j, ells_curr)
            if 'wl' not in tr_i and 'wl' not in tr_j:
                logger.info('No shear tracers in combination. Returning scalar cls.')
                indx = self.saccfile.indices('cl_00', (tr_i, tr_j))
            elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
                logger.info('One shear tracer in combination. Returning scalarxspin2 cls.')
                indx = self.saccfile.indices('cl_0e', (tr_i, tr_j))
            else:
                logger.info('Two shear tracers in combination. Returning spin2 cls.')
                indx = self.saccfile.indices('cl_ee', (tr_i, tr_j))

            cls[indx] = cl_temp

        return cls

    def setup(self, saccfile, ells, hmparams, cosmo):

        logger.info('Setting up GSKYPrediction.')
        if not type(saccfile) == sacc.sacc.Sacc:
            saccfile = sacc.Sacc.load_fits(saccfile)
        self.saccfile = saccfile
        self.ells = ells
        if self.ells == 'NONE':
            logger.info('No ell array provided using probe-specific ells from sacc.')
        self.fid_cosmo = cosmo

        self.gskytheor = GSKYTheory(self.saccfile, hmparams, cosmo)



