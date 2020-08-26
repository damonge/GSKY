import numpy as np
from .gsky_theory import GSKYTheory
import sacc
import pyccl as ccl
import theory.theory_util as tutil
from theory.theory_util import ClInterpolator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSKYPrediction(object):

    def __init__ (self, saccfile, ells='NONE', hmparams=None, cosmo=None, conv_win=False):

        self.setup(saccfile, ells, hmparams, cosmo, conv_win)

    def get_prediction(self, params, trc_combs=None):

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

        if self.ells != 'NONE':
            cls = []
        else:
            cls = np.zeros_like(self.saccfile.mean)

        for tr_i, tr_j in trc_combs:
            logger.info('Computing theory prediction for tracers {}, {}.'.format(tr_i, tr_j))

            if 'wl' not in tr_i and 'wl' not in tr_j:
                logger.info('No shear tracers in combination. Returning scalar cls.')
                datatype = 'cl_00'
            elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
                logger.info('One shear tracer in combination. Returning scalarxspin2 cls.')
                datatype = 'cl_0e'
            else:
                logger.info('Two shear tracers in combination. Returning spin2 cls.')
                datatype = 'cl_ee'

            if self.ells != 'NONE':
                if self.conv_win:
                    # Get window
                    win = self.saccfile.get_tag('window', tracers=(tr_i, tr_j), data_type=datatype)
                    if type(win) is list:
                        win = win[0]
                    ell_max = win.values.shape[0]
                    itp = ClInterpolator(self.ells, np.amax(ell_max))
                    cl_temp = self.gskytheor.getCls(tr_i, tr_j, itp.ls_eval)
                else:
                    cl_temp = self.gskytheor.getCls(tr_i, tr_j, self.ells)

                if self.conv_win:
                    cl_temp = tutil.interp_and_convolve(cl_temp, win, itp)

                cls.append(cl_temp)

            else:
                ells_curr = np.array(self.saccfile.get_tag('ell', tracers=(tr_i, tr_j), data_type=datatype))
                if self.conv_win:
                    # Get window
                    win = self.saccfile.get_tag('window', tracers=(tr_i, tr_j), data_type=datatype)
                    if type(win) is list:
                        win = win[0]
                    ell_max = win.values.shape[0]
                    itp = ClInterpolator(self.ells, np.amax(ell_max))
                    cl_temp = self.gskytheor.getCls(tr_i, tr_j, itp.ls_eval)
                else:
                    cl_temp = self.gskytheor.getCls(tr_i, tr_j, ells_curr)

                indx = self.saccfile.indices(datatype, (tr_i, tr_j))

                if self.conv_win:
                    cl_temp = tutil.interp_and_convolve(cl_temp, win, itp)

                cls[indx] = cl_temp

        return cls

    def setup(self, saccfile, ells, hmparams, cosmo, conv_win):

        logger.info('Setting up GSKYPrediction.')
        if not type(saccfile) == sacc.sacc.Sacc:
            saccfile = sacc.Sacc.load_fits(saccfile)
        self.saccfile = saccfile
        self.ells = ells
        if self.ells == 'NONE':
            logger.info('No ell array provided using probe-specific ells from sacc.')
        self.conv_win = conv_win
        if self.conv_win:
            logger.info('Convolving theory prediction with namaster window functions.')
        else:
            logger.info('Not convolving theory prediction with namaster window functions.')
        self.fid_cosmo = cosmo

        self.gskytheor = GSKYTheory(self.saccfile, hmparams, cosmo)



