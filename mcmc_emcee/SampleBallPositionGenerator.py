import numpy as np


class SampleBallPositionGenerator(object):
    """
        Generates samples in a very tight n-dimensional ball
    """

    def setup(self, params, nwalkers):
        """
            setup the generator
        """
        self.params = params
        self.nwalkers = nwalkers

    def generate(self):
        """
            generates the positions
        """

        p0 = self.params[:, 0] + np.random.normal(size=(self.nwalkers, self.params.shape[0])) * self.params[:, 3][None, :]

        return p0

    def __str__(self, *args, **kwargs):
        return "SampleBallPositionGenerator"