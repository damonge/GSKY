#! /usr/bin/env python

import numpy as np
import scipy.interpolate

class ClInterpolator(object):
    def __init__(self, lb, lmax, mode='interp', nrb=None, nb_dex_extrap_lo=10, kind=None):
        """Interpolator for angular power spectra
        lb : central bandpower ells
        nrb : re-binning factor for ells within the range of the bandpowers
        nb_dex_extrap_lo : number of ells per decade for ells below the range of the bandpowers
        kind : interpolation type

        Extrapolation at high ell will be done assuming a power-law behaviour,
        with a power-law index estimated from the last two elements of the power spectrum.

        Once initialized, ClInterpolator.ls_eval holds the multipole values at which the
        power spectra should be estimated.
        """

        self.mode = mode

        if self.mode == 'interp':
            if nrb is None:
                nrb = 20
            if kind is None:
                kind = 'linear'
        elif self.mode == 'extrap':
            if nrb is None:
                nrb = 20
            if kind is None:
                kind = 'cubic'
        else:
            raise NotImplementedError()

        # Interpolation type
        self.kind = kind

        # Ells below the rannge
        ls_pre=np.geomspace(2, lb[0], nb_dex_extrap_lo*np.log10(lb[0]/2.))
        # Ells in range
        ls_mid=(lb[:-1, None]+(np.arange(nrb)[None,:]*np.diff(lb)[:,None]/nrb)).flatten()[1:]

        if self.mode == 'interp':
            # Ells above range
            ls_post = np.geomspace(lb[-1], 2*lb[-1], 50, endpoint=False)
            ls_extrap = np.geomspace(2*lb[-1], lmax, 20)
            self.ls_eval = np.concatenate((ls_pre, ls_mid, ls_post, ls_extrap))
        elif self.mode == 'extrap':
            # Ells above range
            ls_post = np.geomspace(lb[-1], 2*lb[-1], 50)
            self.ls_eval = np.concatenate((ls_pre, ls_mid, ls_post))


    def interpolate_and_extrapolate(self,ls,clb):
        """Go from a C_ell estimated in a few ells to one estimated in a
        finer grid of ells.

        ls : finer grid of ells
        clb : power spectra evaluated at self.ls_eval

        returns : power spectrum evaluated at ls
        """

        if self.mode == 'interp':
            cli = scipy.interpolate.interp1d(self.ls_eval,clb,kind=self.kind,fill_value=0,bounds_error=False)
            clret = cli(ls)

        elif self.mode == 'extrap':
            # Ells in range
            ind_good = np.where(ls<=self.ls_eval[-1])[0]
            ind_bad = np.where(ls>self.ls_eval[-1])[0]
            clret = np.zeros(len(ls))
            cli = scipy.interpolate.interp1d(self.ls_eval,clb,kind=self.kind,fill_value=0,bounds_error=False)
            clret[ind_good] = cli(ls[ind_good])

            # Extrapolate at high ell
            clret[ind_bad] = clb[-1]*(ls[ind_bad]/self.ls_eval[-1])**-1.05

        return clret


def interp_and_convolve(cl, win, itp):

    weight = win.weight
    ell = win.values
    nbands = weight.shape[0]
    # Extrapolate at high ell
    cls = itp.interpolate_and_extrapolate(ell, cl)

    cl_conv = np.zeros(nbands)
    # Convolve with windows
    for j in range(nbands):
        cl_conv[j] = np.sum(weight[:, j] * cls)

    return cl_conv