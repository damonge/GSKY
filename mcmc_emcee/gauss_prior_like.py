import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GaussLike(object):

    def __init__ (self, mean, cov, paramIndx=None):

        self.mean = mean
        self.invcov = np.linalg.inv(cov)

        self.paramIndx = paramIndx

    def computeLikelihood(self, params):

        # Calculate a likelihood up to normalization
        if self.paramIndx is not None:
            params_trim = params[self.paramIndx]
        else:
            params_trim = params
        delta = self.mean - params_trim
        lnprob = np.einsum('i,ij,j', delta, self.invcov, delta)
        lnprob *= -0.5

        # Return the likelihood
        return lnprob

    def setup(self):

        pass

