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
        cl_temp = gskytheor.getCls(tr_i, tr_j, ells)
        if 'wl' not in tr_i and 'wl' not in tr_j:
            indx = saccfile.indices('cl_00', tr_i, tr_j)
        elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
            indx = saccfile.indices('cl_0e', tr_i, tr_j)
        else:
            indx = saccfile.indices('cl_ee', tr_i, tr_j)

        cls[indx] = cl_temp

    return cls



