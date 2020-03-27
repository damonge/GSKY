import numpy as np
import pyccl as ccl

class ConcentrationDuffy08M500c(ccl.halos.Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.

    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08M500c'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08M500c, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(500, 'critical')

    def _check_mdef(self, mdef):
        if (mdef.Delta != 500) or (mdef.rho_type != 'critical'):
            return True
        return False

    def _setup(self):
        self.A = 3.67
        self.B = -0.0903
        self.C = -0.51

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)
