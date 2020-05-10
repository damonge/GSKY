import numpy as np

class InitializeFromChain(object):
    """
    
        Picks random positions from a given sample to initialize the walkers.
    """
    
    def __init__(self, path, fraction = 0.5):
        """
            default constructor
        """
        self.path = path
        self.fraction = fraction
        
    def setup(self, sampler):
        sample = np.loadtxt(self.path)
        if sample.shape[1] != sampler.paramCount:
            raise Warning('Sample dimensions do not agree with likelihood ones.')
        nmin = int(sample.shape[0]*float(self.fraction))
        self.n = sample.shape[0]-nmin
        self.sample = sample[nmin:]
        self.nwalkers = sampler.nwalkers
        
    def generate(self):
        """
            generates the positions
        """
        # pos = np.random.randint(0, self.n, size=self.nwalkers)
        pos = np.random.choice(self.n, size=self.nwalkers, replace=False)

        print('Initial positions: ', self.sample[pos])
        
        return self.sample[pos]
    
    def __str__(self, *args, **kwargs):
        return "InitializeFromChain"
    