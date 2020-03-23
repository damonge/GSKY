import numpy as np
from gsky_theory import GSKYTheory
import sacc

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prediction(saccfile, params, ells):

    if not type(saccfile) == sacc.sacc.Sacc:
        saccfile = sacc.Sacc.load_fits(saccfile)

    gskytheor = GSKYTheory(saccfile, params)

    cls = np.zeros_like(saccfile.mean)

    for tr_i, tr_j in saccfile.get_tracer_combinations():
        logger.info('Computing theory prediction for tracers {}, {}.'.format(tr_i, tr_j))
        cl_temp = gskytheor.getCls(tr_i, tr_j, ells)
        if 'wl' not in tr_i and 'wl' not in tr_j:
            logger.info('No shear tracers in combination. Returning scalar cls.')
            indx = saccfile.indices('cl_00', (tr_i, tr_j))
        elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
            logger.info('One shear tracer in combination. Returning scalarxspin2 cls.')
            indx = saccfile.indices('cl_0e', (tr_i, tr_j))
        else:
            logger.info('Two shear tracers in combination. Returning spin2 cls.')
            indx = saccfile.indices('cl_ee', (tr_i, tr_j))

        cls[indx] = cl_temp

    return cls



