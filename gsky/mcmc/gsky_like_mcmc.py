import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSKYLike(object):

    def __init__ (self, saccfile, noise_saccfile=None):

        self.obs_data = saccfile.get_mean()
        if noise_saccfile is not None:
            self.obs_data -= noise_saccfile.get_mean()

        self.invcov = np.linalg.inv(saccfile.covariance.covmat)

    def computeLikelihood(self, ctx):

        # Calculate a likelihood up to normalization
        obs_theory = ctx.get('obs_theory')
        delta = self.obs_data - obs_theory
        lnprob = np.einsum('i,ij,j', delta, self.invcov, delta)
        lnprob *= -0.5

        # Return the likelihood
        return lnprob

    def setup(self):

        pass

