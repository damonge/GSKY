import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSKYLike(object):

    def __init__ (self, saccfile, noise_saccfile=None):

        self.obs_data = saccfile.mean
        if noise_saccfile is not None:
            self.obs_data -= noise_saccfile.mean

        # self.invcov = np.linalg.inv(saccfile.covariance.covmat)

        del saccfile
        if noise_saccfile is not None:
            del noise_saccfile

    def computeLikelihood(self, ctx):

        invcov = np.load('/global/cscratch1/sd/anicola/DATA/HSCxACT/HSC/HSC_processed/invcov.npy')

        # Calculate a likelihood up to normalization
        obs_theory = ctx.get('obs_theory')
        delta = self.obs_data - obs_theory
        lnprob = np.einsum('i,ij,j', delta, invcov, delta)
        lnprob *= -0.5

        # Return the likelihood
        return lnprob

    def setup(self):

        pass

