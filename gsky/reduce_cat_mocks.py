from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os
import healpy as hp
import healsparse as hsp
from .flatmaps import FlatMapInfo
from .map_utils import (createCountsMap,
                        createMeanStdMaps,
                        createMask,
                        removeDisconnected,
                        createSpin2Map,
                        createW2QU2Map)
from .estDepth import get_depth
from .estDepth import get_seeing
from .plot_utils import plot_histo, plot_map
from astropy.io import fits
import copy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReduceCatMocks(PipelineStage):
    name = "ReduceCatMocks"
    inputs = [('mock_catalog', FitsFile)]
    outputs = [('clean_catalog', FitsFile),
               ('masked_fraction', FitsFile)]
    # outputs = [('clean_catalog', FitsFile),
    #            ('dust_map', FitsFile),
    #            ('star_map', FitsFile),
    #            ('masked_fraction', FitsFile),
    #            ('depth_map', FitsFile),
    #            ('seeing_map', FitsFile)]
    config_options = {'plots_dir': None,
                      'min_snr': 10., 'depth_cut': 24.5,
                      'mapping': {'wcs': None, 'res': 0.0285,
                                  'res_bo': 0.003, 'pad': 0.1,
                                  'projection': 'CAR'},
                      'band': 'i', 'depth_method': 'fluxerr',
                      'shearrot': 'noflip', 'mask_type': 'sirius',
                      'ra':  'ra_mock', 'dec':  'dec_mock',
                      'pz_code': 'dnnz', 'pz_mark': 'best',
                      'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5]}
    bands = ['g', 'r', 'i', 'z', 'y']

    def make_dust_map(self, cat, fsk):
        """
        Produces a dust absorption map for each band.
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry
            of the output map
        """
        logger.info("Creating dust map")
        dustmaps = []
        dustdesc = []
        for b in self.bands:
            m, s = createMeanStdMaps(cat[self.config['ra']],
                                     cat[self.config['dec']],
                                     cat['a_'+b], fsk)
            dustmaps.append(m)
            dustdesc.append('Dust, '+b+'-band')
        return dustmaps, dustdesc

    def make_star_map(self, cat, fsk, sel):
        """
        Produces a star density map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry
            of the output map
        :param sel: mask used to select the stars to be used.
        """
        logger.info("Creating star map")
        # mstar = createCountsMap(cat[self.config['ra']][sel],
        #                         cat[self.config['dec']][sel],
        #                         fsk)+0.
        mstar = createCountsMap(cat[self.config['ra']],
                                cat[self.config['dec']],
                                fsk)
        descstar = ('Stars, '+self.config['band'] +
                    '<%.2lf' % (self.config['depth_cut']))
        return mstar, descstar

    def make_bo_mask(self, cat, fsk, mask_fulldepth=False):
        """
        Produces a bright object mask
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the
            geometry of the output map
        """
        logger.info("Generating bright-object mask")
        if self.config['mask_type'] == 'arcturus':
            flags_mask = [~cat['mask_Arcturus'].astype(bool)]
        elif self.config['mask_type'] == 'sirius':
            flags_mask = [cat['iflags_pixel_bright_object_center'],
                          cat['iflags_pixel_bright_object_any']]
        else:
            raise ValueError('Mask type '+self.config['mask_type'] +
                             ' not supported')
        if mask_fulldepth:
            flags_mask.append(~cat['wl_fulldepth_fullcolor'].astype(bool))
        mask_bo, fsg = createMask(cat[self.config['ra']],
                                  cat[self.config['dec']],
                                  flags_mask, fsk,
                                  self.mpp['res_bo'])
        return mask_bo, fsg

    def make_masked_fraction(self, cat, fsk, mask_fulldepth=False):
        """
        Produces a masked fraction map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the
            geometry of the output map
        """
        logger.info("Generating masked fraction map")

        if 'VVDS' in self.get_input('mock_catalog'):
            print("Max and Min RA", np.max(cat[self.config['ra']]), np.min(cat[self.config['ra']]))
            print("Shifting RA back by +30 degrees for VVDS for FDFC cut")
            change_in_ra = +30.0
            init_ra_vals = cat[self.config['ra']].copy()
            reshifted_ra_vals = cat[self.config['ra']].copy()
            reshifted_ra_vals = init_ra_vals+(np.ones(len(init_ra_vals))*change_in_ra)
            reshifted_ra_vals[reshifted_ra_vals>360] -= 360.0
            print("Max and Min RA", np.max(reshifted_ra_vals), np.min(reshifted_ra_vals))


        masked = np.ones(len(cat))
        # full depth full color cut based on healpix map
        hpfname =   "/tigress/rdalal/s19a_shear/s19a_fdfc_hp_contarea_izy-gt-5_trimmed_fd001.fits"
        # hpfname = "/tigress/rdalal/s19a_shear/shared_frames/final_fdfc_map.hs"
        m       =   hp.read_map(hpfname, nest = True, dtype = np.bool)
        # m_hsp       =   hsp.HealSparseMap.read(hpfname)
        # m = m.generate_healpix_map(nside=16384, reduction='mean')
        mfactor =   np.pi/180.
        indices_map =   np.where(m)[0]
        # indices_map =   np.where(m[m.valid_pixels])[0]
        nside   =   hp.get_nside(m)
        # nside   =   m.nside_sparse
        # print("nside", nside)
        if 'VVDS' in self.get_input('mock_catalog'):
            phi     =   reshifted_ra_vals*mfactor
        else:
            phi     =   cat[self.config['ra']]*mfactor
        theta   =   np.pi/2. - cat[self.config['dec']]*mfactor
        indices_obj = hp.ang2pix(nside, theta, phi, nest = True)
        print("masked sum", np.sum(masked))
        # masked *= np.in1d(indices_obj, indices_map)
        # print("masked sum", np.sum(masked))

        # bright object mask
        # masked *= np.logical_not(cat['i_mask_brightstar_ghost15'])
        # masked *= np.logical_not(cat['i_mask_brightstar_halo'])
        # masked *= np.logical_not(cat['i_mask_brightstar_blooming'])
        # if mask_fulldepth:
        #     masked *= cat['wl_fulldepth_fullcolor']
        # if self.config['mask_type'] == 'arcturus':
        #     masked *= cat['mask_Arcturus']
        # elif self.config['mask_type'] == 'sirius':
        #     masked *= np.logical_not(cat['iflags_pixel_bright_object_center'])
        #     masked *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        # else:
        #     raise ValueError('Mask type '+self.config['mask_type'] +
        #                      ' not supported')
        masked_fraction, _ = createMeanStdMaps(cat[self.config['ra']],
                                               cat[self.config['dec']],
                                               masked, fsk)
        masked_fraction_cont = removeDisconnected(masked_fraction, fsk)
        return masked_fraction_cont

    def make_depth_map(self, cat, fsk):
        """
        Produces a depth map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the
            geometry of the output map
        """
        logger.info("Creating depth maps")
        method = self.config['depth_method']
        band = self.config['band']
        snrs = cat['%s_psfflux_flux' % band]/cat['%s_psfflux_fluxerr' % band]
        if method == 'fluxerr':
            arr1 = cat['%s_psfflux_fluxerr' % band]
            arr1_copy = arr1
            print('min arr1', np.min(arr1_copy))
            # convert to erg s^{-1} cm^{-2} Hz^{-1}
            arr1_copy = np.array(arr1_copy*pow(10, -32), dtype='float64')
            arr2 = None
        else:
            arr1 = cat['%s_psfflux_mag' % band]
            arr1_copy = np.copy(arr1)
            arr2 = snrs
        print(arr1)    
        depth, _ = get_depth(method, cat[self.config['ra']][cat['i_psfflux_mag']>21.5],
                             cat[self.config['dec']][cat['i_psfflux_mag']>21.5],
                             arr1=arr1_copy[cat['i_psfflux_mag']>21.5], arr2=arr2,
                             fsk=fsk, snrthreshold=self.config['min_snr'],
                             interpolate=True, count_threshold=4)
        desc = '%d-s depth, ' % (self.config['min_snr'])+band+' '+method+' mean'

        return depth, desc

    def make_seeing_map(self, cat, fsk):
        """
        Produces a seeing map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the
            geometry of the output map
        """
        logger.info("Creating seeing maps")
        #method = self.config['depth_method']
        band = self.config['band']
        # good_object_id = np.ones(len(cat))
        # good_object_id *= np.logical_not(cat['i_mask_brightstar_ghost'])
        # psf_11 = cat['i_sdssshape_shape11'][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22']))]
        # psf_22 = cat['i_sdssshape_shape22'][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22']))]
        psf_11 = cat['i_sdssshape_shape11']
        psf_22 = cat['i_sdssshape_shape22']
        print("psf_11", np.min(psf_11), np.max(psf_11))
        print(np.sum(np.isnan(psf_11)))
        print("psf_22", np.min(psf_22), np.max(psf_22))
        print(np.sum(np.isnan(psf_22)))
        arr1 = np.sqrt(0.5*(psf_11+psf_22))
        print(np.min(arr1), np.max(arr1))
        print("Mean seeing", np.mean(arr1))
        seeing, _ = get_seeing(cat[self.config['ra']][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22'])) & np.logical_not(arr1>5.0)],
                             cat[self.config['dec']][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22'])) & np.logical_not(arr1>5.0)],
                             arr1=arr1[np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22'])) & np.logical_not(arr1>5.0)],
                             fsk=fsk,
                             interpolate=True, count_threshold=4)
        desc = '%d-s seeing, ' % (self.config['min_snr'])+band+' '+' mean'

        return seeing, desc

    def match_star_cats(self, cat, sel, star_cat):
        """
        Cross-match the non-PSF stars to the star catalog provided by Eli
        https://lsst.ncsa.illinois.edu/~erykoff/PDR2/fgcm_standard_stars/DM-23243/fgcmStandardStars-05.fits
        :param cat:
        :param sel:
        :param star_cat:
        :return:
        """

        logger.info('Matching star catalogs.')

        star_cat_sel = copy.deepcopy(cat)[sel]
        logger.info('Initial size of star catalog selected from HSC = {}.'.format(len(star_cat_sel)))
        star_cat_coord = SkyCoord(ra=star_cat['coord_ra'] * u.radian, dec=star_cat['coord_dec'] * u.radian)

        star_cat_sel_coord = SkyCoord(ra=star_cat_sel['ra'] * u.degree, dec=star_cat_sel['dec'] * u.degree)
        _, d2d, _ = star_cat_sel_coord.match_to_catalog_sky(star_cat_coord)

        max_sep = 1.0 * u.arcsec
        sep_constraint = d2d < max_sep
        star_cat_sel_matched = star_cat_sel[sep_constraint]
        logger.info('Size of HSC star catalog cross-matched to external star catalog = {}.'.format(len(star_cat_sel_matched)))

        return star_cat_sel_matched

    def make_PSF_maps(self, star_cat, fsk):
        """
        Get e_PSF, 1, e_PSF, 2, T_PSF maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # PSF of stars
        Mxx = star_cat['i_sdssshape_psf_shape11']
        Myy = star_cat['i_sdssshape_psf_shape22']
        Mxy = star_cat['i_sdssshape_psf_shape12']
        T_I = Mxx + Myy
        e_plus_I = (Mxx - Myy)/T_I
        e_cross_I = 2*Mxy/T_I
        ePSFmaps, ePSFmasks = createSpin2Map(star_cat[self.config['ra']],
                                             star_cat[self.config['dec']],
                                             e_plus_I, e_cross_I, fsk,
                                             shearrot=self.config['shearrot'])

        TPSFmap, _ = createMeanStdMaps(star_cat[self.config['ra']],
                                             star_cat[self.config['dec']],
                                             T_I, fsk)

        maps = [ePSFmaps, ePSFmasks, TPSFmap]

        return maps, e_plus_I, e_cross_I, T_I

    def make_PSF_res_maps(self, star_cat, fsk):
        """
        Get e_PSF, 1, e_PSF, 2, T_PSF residual maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # PSF of stars
        Mxx = star_cat['i_sdssshape_psf_shape11']
        Myy = star_cat['i_sdssshape_psf_shape22']
        Mxy = star_cat['i_sdssshape_psf_shape12']
        T_PSF = Mxx + Myy
        e_plus_PSF = (Mxx - Myy)/T_PSF
        e_cross_PSF = 2*Mxy/T_PSF

        Mxx = star_cat['i_sdssshape_shape11']
        Myy = star_cat['i_sdssshape_shape22']
        Mxy = star_cat['i_sdssshape_shape12']
        T_I = Mxx + Myy
        e_plus_I = (Mxx - Myy)/T_I
        e_cross_I = 2*Mxy/T_I

        delta_e_plus = e_plus_PSF - e_plus_I
        delta_e_cross = e_cross_PSF - e_cross_I

        ePSFresmaps, ePSFresmasks = createSpin2Map(star_cat[self.config['ra']],
                                                   star_cat[self.config['dec']],
                                                   delta_e_plus, delta_e_cross, fsk,
                                                   shearrot=self.config['shearrot'])

        delta_T = T_PSF - T_I

        TPSFresmap, _ = createMeanStdMaps(star_cat[self.config['ra']], star_cat[self.config['dec']],
                                                   delta_T, fsk)

        maps = [ePSFresmaps, ePSFresmasks, TPSFresmap]

        return maps, delta_e_plus, delta_e_cross, delta_T, e_plus_I, e_cross_I

    def shear_cut(self, cat):
        """
        Apply additional shear cuts to catalog.
        :param cat:
        :return:
        """

        logger.info('Applying shear cuts to catalog.')

        # ishape_flags_mask = ~cat['ishape_hsm_regauss_flags']
        # ishape_sigma_mask = ~np.isnan(cat['ishape_hsm_regauss_sigma'])
        # ishape_resolution_mask = cat['ishape_hsm_regauss_resolution'] >= 0.3
        # ishape_shear_mod_mask = (cat['ishape_hsm_regauss_e1']**2 + cat['ishape_hsm_regauss_e2']**2) < 2
        # ishape_sigma_mask *= (cat['ishape_hsm_regauss_sigma'] >= 0.)*(cat['ishape_hsm_regauss_sigma'] <= 0.4)
        # # Remove masked objects
        # if self.config['mask_type'] == 'arcturus':
        #     star_mask = cat['mask_Arcturus'].astype(bool)
        # elif self.config['mask_type'] == 'sirius':
        #     star_mask = np.logical_not(cat['iflags_pixel_bright_object_center'])
        #     star_mask *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        # else:
        #     raise KeyError("Mask type "+self.config['mask_type'] +
        #                    " not supported. Choose arcturus or sirius")
        # fdfc_mask = cat['wl_fulldepth_fullcolor']

        # shearmask = ishape_flags_mask*ishape_sigma_mask*ishape_resolution_mask*ishape_shear_mod_mask*star_mask*fdfc_mask
        
        return np.ones(len(cat[self.config['ra']]))

    def shear_calibrate(self, cat):
        # Galaxies used for shear
        # mask_shear = cat['shear_cat'] & (cat['tomo_bin'] >= 0)
        mask_shear = cat['tomo_bin'] >= 0

        # Calibrate shears per redshift bin
        e1cal = np.zeros(len(cat))
        e2cal = np.zeros(len(cat))
        mhats = np.zeros(self.nbins)
        resps = np.zeros(self.nbins)
        for ibin in range(self.nbins):
            mask_bin = mask_shear & (cat['tomo_bin'] == ibin)
            # Compute multiplicative bias
            mhat = np.average(cat[mask_bin]['i_hsmshaperegauss_derived_shear_bias_m'],
                              weights=cat[mask_bin]['i_hsmshaperegauss_derived_weight'])
            mhats[ibin] = mhat
            # Compute responsivity
            resp = 1. - np.average(cat[mask_bin]['i_hsmshaperegauss_derived_rms_e'] ** 2,
                                   weights=cat[mask_bin]['i_hsmshaperegauss_derived_weight'])
            resps[ibin] = resp

            e1 = (cat[mask_bin]['i_hsmshaperegauss_e1']/(2.*resp) -
                  cat[mask_bin]['i_hsmshaperegauss_derived_shear_bias_c1']) / (1 + mhat)
            e2 = (cat[mask_bin]['i_hsmshaperegauss_e2']/(2.*resp) -
                  cat[mask_bin]['i_hsmshaperegauss_derived_shear_bias_c2']) / (1 + mhat)
            e1cal[mask_bin] = e1
            e2cal[mask_bin] = e2
        return e1cal, e2cal, mhats, resps

    def get_w2e2(self, cat, e1, e2, fsk):
        """
        Compute the weighted mean squared ellipticity in a pixel, averaged over the whole map (used for analytic shape
        noise estimation).
        :param cat:
        :return:
        """

        w2e2maps = createW2QU2Map(cat[self.config['ra']],
                                  cat[self.config['dec']],
                                  e1,
                                  e2, fsk,
                                  weights=None)

        w2e2 = 0.5*(np.mean(w2e2maps[0]) + np.mean(w2e2maps[1]))

        return w2e2

    def pz_binning(self, cat):
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][1:]
        self.nbins = len(zi_arr)

        # if self.config['pz_code'] == 'dnnz':
        #     self.pz_code = 'dnnz'
        # # elif self.config['pz_code'] == 'frankenz':
        # #     self.pz_code = 'frz'
        # # elif self.config['pz_code'] == 'nnpz':
        # #     self.pz_code = 'nnz'
        # else:
        #     raise KeyError("Photo-z method "+self.config['pz_code'] +
        #                    " unavailable. Choose dnnz")

        # if self.config['pz_mark'] not in ['best', 'mean', 'mode', 'mc']:
        #     raise KeyError("Photo-z mark "+self.config['pz_mark'] +
        #                    " unavailable. Choose between "
        #                    "best, mean, mode and mc")

        # self.column_mark = 'pz_'+self.config['pz_mark']+'_'+self.pz_code
        # self.column_mark = self.pz_code+'_photoz_'+self.config['pz_mark']
        zs = cat['z_source_mock']

        # Assign all galaxies to bin -1
        bin_number = np.zeros(len(cat), dtype=int) - 1

        for ib, (zi, zf) in enumerate(zip(zi_arr, zf_arr)):
            msk = (zs <= zf) & (zs > zi)
            bin_number[msk] = ib
        return bin_number

    def get_abs_ellip(self, catalog):
    #Returns the modulus of galaxy distortions.
        if 'absE' in catalog.dtype.names:
            absE    =   catalog['absE']
        elif 'i_hsmshaperegauss_e1' in catalog.dtype.names:# For S18A
            absE    =   catalog['i_hsmshaperegauss_e1']**2.+catalog['i_hsmshaperegauss_e2']**2.
            absE    =   np.sqrt(absE)
        elif 'ishape_hsm_regauss_e1' in catalog.dtype.names:# For S16A
            absE    =   catalog['ishape_hsm_regauss_e1']**2.+catalog['ishape_hsm_regauss_e2']**2.
            absE    =   np.sqrt(absE)
        elif 'ext_shapeHSM_HsmShapeRegauss_e1' in catalog.dtype.names:# For pipe 7
            absE    =   catalog['ext_shapeHSM_HsmShapeRegauss_e1']**2.\
                        +catalog['ext_shapeHSM_HsmShapeRegauss_e2']**2.
            absE    =   np.sqrt(absE)
        else:
            absE  =   np.empty(len(catalog))
            absE.fill(np.nan)
        return absE

    def get_sdss_size(self, catalog,type='det'):
        """
        This utility gets the observed galaxy size from a data or sims catalog using the
        specified size definition from the second moments matrix.

        Parameters:
            catalog: recarray
                Simulation or data catalog
            type: string
                Type of psf size measurement in ['trace', 'determin']

        Returns:
            gal_size [arcsec]
        """
        if 'base_SdssShape_xx' in catalog.dtype.names:        #pipe 7
            gal_mxx = catalog['base_SdssShape_xx']*0.168**2.
            gal_myy = catalog['base_SdssShape_yy']*0.168**2.
            gal_mxy = catalog['base_SdssShape_xy']*0.168**2.
        elif 'i_sdssshape_shape11' in catalog.dtype.names:  #s18 & s19
            gal_mxx = catalog['i_sdssshape_shape11']
            gal_myy = catalog['i_sdssshape_shape22']
            gal_mxy = catalog['i_sdssshape_shape12']
        elif 'ishape_sdss_ixx' in catalog.dtype.names:      #s15
            gal_mxx = catalog['ishape_sdss_ixx']
            gal_myy = catalog['ishape_sdss_iyy']
            gal_mxy = catalog['ishape_sdss_ixy']
        else:
            gal_mxx  =   np.empty(len(catalog))
            gal_mxx.fill(np.nan)
            gal_myy  =   np.empty(len(catalog))
            gal_myy.fill(np.nan)
            gal_mxy  =   np.empty(len(catalog))
            gal_mxy.fill(np.nan)

        if type == 'trace':
            size = np.sqrt(gal_mxx + gal_myy)
        elif type == 'det':
            size = (gal_mxx * gal_myy - gal_mxy**2)**(0.25)
        else:
            raise ValueError("Unknown PSF size type: %s"%type)
        return size

    def get_binarystar_flags(self, data):
        """
        Get the flags for binary stars (|e|>0.8 & logR<1.8-0.1r)
        Parameters:
            an hsc-like catalog
            [ndarray,table]
        Returns:
            a boolean (True for binary stars)
        """
        absE=   self.get_abs_ellip(data)
        logR=   np.log10(self.get_sdss_size(data))
        rmag=   data['forced_r_cmodel_mag']-data['a_r']
        msk =   absE>0.8
        a=1;b=10.;c=-18.
        msk =   msk & ((a*rmag+b*logR+c)<0.)
        return msk

    def get_sel_bias(self, weight, magA10, res):
        """
        This utility gets the selection bias (multiplicative and additive)
        Parameters:
            weight: array_like
                Weight for dataset.  E.g., lensing shape weight, Sigma_c^-2 weight
            res: array_like
                Resolution factor for dataset
            magA10: array_like
                aperture magnitude (1 arcsec) for dataset
        Returns:
            m_sel (float) :
                Multiplicative edge-selection bias
            a_sel (float) :
                additive edge-selection bias (c1)
            m_sel_err (float) :
                1-sigma uncertainty in m_sel
            a_sel_err (float) :
                1-sigma uncertainty in a_sel
        """

        if not(np.all(np.isfinite(weight))):
            raise ValueError("Non-finite weight")
        if not(np.all(weight) >= 0.0):
            raise ValueError("Negative weight")
        wSum    =   np.sum(weight)

        bin_magA=   0.025
        pedgeM  =   np.sum(weight[(magA10>= 25.5-bin_magA)])/wSum/bin_magA

        bin_res =   0.01
        pedgeR  =   np.sum(weight[(res<= 0.3+bin_res)])/wSum/bin_res

        m_sel   =   -0.059*pedgeM+0.019*pedgeR
        a_sel   =   0.0064*pedgeM+0.0063*pedgeR

        # assume the errors for 2 cuts are independent.
        m_err   =   np.sqrt((0.0089*pedgeM)**2.+(0.0013*pedgeR)**2.)
        a_err   =   np.sqrt((0.0034*pedgeM)**2.+(0.0009*pedgeR)**2.)
        return m_sel,a_sel,m_err,a_err

    def add_mbias(self, datIn, mbias, msel, corr):
        """
        Rescale the shear by (1 + mbias) following section 5.6 and calculate the
        mock ellipticities according to eq. (24) and (25) of
        https://arxiv.org/pdf/1901.09488.pdf
        Args:
            datIn (ndaray): Original HSC S19A mock catalog (it should haave m=0)
            mbias (float):  The multiplicative bias
            msel (float):   Selection bias [default=0.]
            corr (float):   Correction term for shell thickness, finite resolution and missmatch
                            between n(z_data) and n(z_mock) due to a limited number source planes
        Returns:
            out (ndarray):  Updated S19A mock catalog (with m=mbias)
        """


        # if not isinstance(mbias,(float,int)):
        #     raise TypeError('multiplicative shear estimation bias should be a float.')
        # if not isinstance(msel,(float,int)):
        #     raise TypeError('multiplicative selection bias should be a float.')
        bratio_arr = np.ones(self.nbins+1)
        for ibin in self.bin_indxs:
            bratio_arr[ibin] = (1+mbias[ibin])*(1+msel[ibin])*corr[ibin]
        logger.info('bratios: %f %f %f %f %f' % (bratio_arr[0], bratio_arr[1], bratio_arr[2], bratio_arr[3], bratio_arr[4]))
        out   =  datIn.copy()
        # Rescaled gamma by (1+m) and then calculate the distortion delta
        gamma_sq=(out['shear1_sim']**2.+out['shear2_sim']**2.)*bratio[out['tomo_bin']]**2.
        dis1  =  2.*(1-out['kappa'])*out['shear1_sim']*bratio[out['tomo_bin']]/\
                    ((1-out['kappa'])**2+gamma_sq)
        dis2  =  2.*(1-out['kappa'])*out['shear2_sim']*bratio[out['tomo_bin']]/\
                    ((1-out['kappa'])**2+gamma_sq)
        # Calculate the mock ellitpicities
        de    =  dis1*out['noise1_int']+dis2*out['noise2_int'] # for denominators
        dd    =  dis1**2+dis2**2.
        # avoid dividing by zero (this term is 0 under the limit dd->0)
        tmp1  =  np.divide(dis1,dd,out=np.zeros_like(dd),where=dd!=0)
        tmp2  =  np.divide(dis2,dd,out=np.zeros_like(dd),where=dd!=0)
        # the nominator for e1
        e1_mock= out['noise1_int']+dis1+tmp2*(1-(1-dd)**0.5)*\
            (dis1*out['noise2_int']-dis2*out['noise1_int'])
        # the nominator for e2
        e2_mock= out['noise2_int']+dis2+tmp1*(1-(1-dd)**0.5)*\
            (dis2*out['noise1_int']-dis1*out['noise2_int'])
        # update e1_mock and e2_mock
        out['e1_mock']=e1_mock/(1.+de)+out['noise1_mea']
        out['e2_mock']=e2_mock/(1.+de)+out['noise2_mea']
        return out

    def run(self):
        """
        Main function.
        This stage:
        - Reduces the raw catalog by imposing quality cuts, a cut
          on i-band magnitude and a star-galaxy separation cat.
        - Produces mask maps, dust maps, depth maps and star density maps.
        """
        band = self.config['band']
        self.mpp = self.config['mapping']

        # Read catalog
        # cat = Table.read('/tigress/rdalal/s19a_shear/WIDE12H_no_m.fits')
        cat = Table.read(self.get_input('mock_catalog'))

        if band not in self.bands:
            raise ValueError("Band "+band+" not available")

        logger.info('Initial catalog size: %d' % (len(cat)))

        # Clean nans in ra and dec
        # logger.info("Remove bad visit")
        # sel = np.ones(len(cat), dtype=bool)
        # isnull_names = []
        # for key in cat.keys():
        #     if key.__contains__('isnull'):
        #         if not key.startswith('i_hsmshape') and (not key.startswith('i_sdssshape')):
        #             sel[cat[key]] = 0
        #         isnull_names.append(key)
        #     else:
        #         # Keep photo-zs and shapes even if they're NaNs
        #         if (not key.startswith("photoz_")) and (not key.startswith('i_hsmshape')) and (not key.startswith('i_sdssshape')):
        #             sel[np.isnan(cat[key])] = 0
        # logger.info("Will drop %d rows" % (len(sel)-np.sum(sel)))
        # cat.remove_columns(isnull_names)

        # remove bad visit
        # ra=cat[self.config['ra']]
        # dec=cat[self.config['dec']]
        # def _calDistanceAngle(a1, d1):
        #     a2=130.43
        #     d2=-1.02
        #     a1_f64 = np.array(a1, dtype = np.float64)*np.pi/180.
        #     d1_f64 = np.array(d1, dtype = np.float64)*np.pi/180.
        #     a2_f64 = np.array(a2, dtype = np.float64)*np.pi/180.
        #     d2_f64 = np.array(d2, dtype = np.float64)*np.pi/180.
        #     return np.arccos(np.cos(d1_f64)*np.cos(d2_f64)*np.cos(a1_f64-a2_f64)+np.sin(d1_f64)*np.sin(d2_f64))/np.pi*180.
        # d=_calDistanceAngle(ra,dec)
        # mask_bad_visit1=(ra>130.5)&(ra<131.5)&(dec<-1.5) # disconnected regions
        # mask_bad_visit = (d>0.80)&(~mask_bad_visit1)
        # print("testing")
        # print("Bad visit removal ", np.sum(~mask_bad_visit))
        # print("testing2")
        # cat.remove_rows(~mask_bad_visit)


        logger.info("Basic cleanup of raw catalog")
        sel_raw = np.ones(len(cat), dtype=bool)
        print("Initial size", len(cat))
        # sel_raw *= cat['weak_lensing_flag']
        # print("After WL flag", np.sum(sel_raw))
        # sel_raw *= np.logical_not(cat['i_apertureflux_10_mag']>25.5)
        # print("After aperture mag cut", np.sum(sel_raw))
        # sel_raw *= np.logical_not(cat['i_blendedness_abs']>=pow(10, -0.38))
        # print("After blendedness cut", np.sum(sel_raw))
        # sel_raw *= np.logical_not(np.isnan(cat['i_hsmshaperegauss_sigma']))
        # print("After i_hsmshaperegauss_sigma cut", np.sum(sel_raw))
        # sel_raw *= np.logical_not(cat['i_mask_brightstar_ghost15'])
        # sel_raw *= np.logical_not(cat['i_mask_brightstar_halo'])
        # sel_raw *= np.logical_not(cat['i_mask_brightstar_blooming'])
        # print("After bright object mask", np.sum(sel_raw))
        # hpfname =   "/tigress/rdalal/s19a_shear/s19a_fdfc_hp_contarea_izy-gt-5_trimmed_fd001.fits"
        # # hpfname = "/tigress/rdalal/s19a_shear/shared_frames/final_fdfc_map_psf_cut.hs"
        # m       =   hp.read_map(hpfname, nest = True, dtype = np.bool)
        # # m       =   hsp.HealSparseMap.read(hpfname)
        # mfactor =   np.pi/180.
        # indices_map =   np.where(m)[0]
        # # indices_map =   np.where(m[m.valid_pixels])[0]
        # nside   =   hp.get_nside(m)
        # # nside   =   m.nside_sparse
        # phi     =   cat[self.config['ra']]*mfactor
        # theta   =   np.pi/2. - cat[self.config['dec']]*mfactor
        # indices_obj = hp.ang2pix(nside, theta, phi, nest = True)
        # sel_raw *= np.in1d(indices_obj, indices_map)
        # print("after FDFC cut", np.sum(sel_raw))

        # Collect sample cuts
        #sel_area = cat['wl_fulldepth_fullcolor']
        # sel_clean = cat['clean_photometry']
        sel_area = np.ones(len(cat), dtype=bool)
        sel_clean = np.ones(len(cat), dtype=bool)
        sel_maglim = np.ones(len(cat), dtype=bool)
        # sel_maglim[cat['%s_cmodel_mag' % band] -
        #            cat['a_%s' % band] > self.config['depth_cut']] = 0
        # print("depth cut removes ", len(sel_maglim)-np.sum(sel_maglim))
        # Blending
        sel_blended = np.ones(len(cat), dtype=bool)
        # abs_flux<10^-0.375
        # sel_blended[cat['iblendedness_abs_flux'] >= 0.42169650342] = 0
        # S/N in i
        sel_fluxcut_i = np.ones(len(cat), dtype=bool)
        # sel_fluxcut_i[cat['i_cmodel_flux'] < 10*cat['i_cmodel_fluxerr']] = 0
        # print("S/N cut removes ", len(sel_fluxcut_i)-np.sum(sel_fluxcut_i))
        # S/N in g
        sel_fluxcut_g = np.ones(len(cat), dtype=int)
        # sel_fluxcut_g[cat['gcmodel_flux'] < 5*cat['gcmodel_flux_err']] = 0
        # S/N in r
        sel_fluxcut_r = np.ones(len(cat), dtype=int)
        # sel_fluxcut_r[cat['rcmodel_flux'] < 5*cat['rcmodel_flux_err']] = 0
        # S/N in z
        sel_fluxcut_z = np.ones(len(cat), dtype=int)
        # sel_fluxcut_z[cat['zcmodel_flux'] < 5*cat['zcmodel_flux_err']] = 0
        # S/N in y
        sel_fluxcut_y = np.ones(len(cat), dtype=int)
        # sel_fluxcut_y[cat['ycmodel_flux'] < 5*cat['ycmodel_flux_err']] = 0
        # S/N in grzy (at least 2 pass)
        sel_fluxcut_grzy = (sel_fluxcut_g+sel_fluxcut_r +
                            sel_fluxcut_z+sel_fluxcut_y >= 2)
        # Overall S/N
        sel_fluxcut = sel_fluxcut_i*sel_fluxcut_grzy
        # Stars
        sel_stars = np.ones(len(cat), dtype=bool)
        # sel_stars[cat['iclassification_extendedness'] > 0.99] = 0
        # Galaxies
        sel_gals = np.ones(len(cat), dtype=bool)
        # sel_gals[cat['iclassification_extendedness'] < 0.99] = 0
        # PSF validation set stars
        sel_psf_valid = np.ones(len(cat), dtype=bool)
        # sel_psf_valid[cat['icalib_psf_used'] == True] = 0

        ####

        # Roohi: move VVDS RAs to be on same side of 0 degrees
        if 'VVDS' in self.get_input('mock_catalog'):
            print("Max and Min RA", np.max(cat[self.config['ra']]), np.min(cat[self.config['ra']]))
            # np.savez('/tigress/rdalal/s19a_shear/GSKY_outputs/VVDS_ceci/initial_ras', cat[self.config['ra']])
            print("Shifting RA by -30 degrees for VVDS")
            change_in_ra = -30.0
            init_ra_vals = cat[self.config['ra']].copy()
            cat[self.config['ra']] = init_ra_vals+(np.ones(len(init_ra_vals))*change_in_ra)
            cat[self.config['ra']][cat[self.config['ra']]<0] += 360.0
            # np.savez('/tigress/rdalal/s19a_shear/GSKY_outputs/VVDS_ceci/shifted_ras', cat[self.config['ra']])
            print("Max and Min RA", np.max(cat[self.config['ra']]), np.min(cat[self.config['ra']]))

        # Generate sky projection
        fsk = FlatMapInfo.from_coords(cat[self.config['ra']],
                                      cat[self.config['dec']],
                                      self.mpp)

        ####
        # Generate systematics maps
        # 1- Dust
        # dustmaps, dustdesc = self.make_dust_map(cat, fsk)
        # fsk.write_flat_map(self.get_output('dust_map'), np.array(dustmaps),
        #                    descript=dustdesc)

        # 2- Nstar
        #    This needs to be done for stars passing the same cuts as the
        #    sample (except for the s/g separator)
        # Above magnitude limit

        # if self.get_input('star_catalog') != 'NONE':
        #     logger.info('Reading star catalog from {}.'.format(self.get_input('star_catalog')))
        #     star_cat = Table.read(self.get_input('star_catalog'))
        #     # Roohi: move VVDS RAs to be on same side of 0 degrees
        #     if 'VVDS' in self.get_input('mock_catalog'):
        #         print("Shifting star catalog RA by -30 degrees for VVDS")
        #         print("Max and Min RA", np.max(star_cat[self.config['ra']]), np.min(star_cat[self.config['ra']]))
        #         change_in_ra = -30.0
        #         init_ra_vals = star_cat[self.config['ra']].copy()
        #         star_cat[self.config['ra']] = init_ra_vals+(np.ones(len(init_ra_vals))*change_in_ra)
        #         star_cat[self.config['ra']][star_cat[self.config['ra']]<0] += 360.0
        #         print("Max and Min RA", np.max(star_cat[self.config['ra']]), np.min(star_cat[self.config['ra']]))

        # mstar, descstar = self.make_star_map(star_cat, fsk,
        #                                      sel_clean *
        #                                      sel_maglim *
        #                                      sel_stars *
        #                                      sel_fluxcut *
        #                                      sel_blended)
        # fsk.write_flat_map(self.get_output('star_map'), mstar,
        #                    descript=descstar)

        # 3- e_PSF
        # if self.get_input('star_catalog') != 'NONE':
        #     logger.info('Reading star catalog from {}.'.format(self.get_input('star_catalog')))
        #     star_cat = Table.read(self.get_input('star_catalog'))
        #     # TODO: do these stars need to have the same cuts as our sample?
        #     # star_cat_matched = self.match_star_cats(cat, sel_clean*sel_psf_valid*sel_stars, star_cat)
        #     logger.info('Creating e_PSF and T_PSF maps.')
        #     mPSFstar, e_plus_I, e_cross_I, T_I = self.make_PSF_maps(star_cat, fsk)
        #     logger.info("Computing w2e2.")
        #     w2e2 = self.get_w2e2(star_cat, e_plus_I, e_cross_I, fsk)
        #     logger.info("Writing output to {}.".format(self.get_output('ePSF_map')))
        #     header = fsk.wcs.to_header()
        #     hdus = []
        #     shp_mp = [fsk.ny, fsk.nx]
        #     # Maps
        #     head = header.copy()
        #     head['DESCR'] = ('e_PSF1', 'Description')
        #     hdu = fits.PrimaryHDU(data=mPSFstar[0][0].reshape(shp_mp),
        #                               header=head)
        #     hdus.append(hdu)
        #     head = header.copy()
        #     head['DESCR'] = ('e_PSF2', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFstar[0][1].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     head = header.copy()
        #     head['DESCR'] = ('e_PSF weight mask', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFstar[1][0].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     head['DESCR'] = ('e_PSF binary mask', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFstar[1][1].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     head['DESCR'] = ('counts map (PSF star sample)', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFstar[1][2].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     # w2e2
        #     cols = [fits.Column(name='w2e2', array=np.atleast_1d(w2e2), format='E')]
        #     hdus.append(fits.BinTableHDU.from_columns(cols))
        #     hdulist = fits.HDUList(hdus)
        #     hdulist.writeto(self.get_output('ePSF_map'), overwrite=True)

        #     fsk.write_flat_map(self.get_output('TPSF_map'),
        #                        np.array([mPSFstar[2], mPSFstar[2].astype('bool').astype('int')]),
        #                        descript=['T_PSF', 'T_PSF binary mask'])
        #     star_cat['i_hsmshape_PSF_e1'] = e_plus_I
        #     star_cat['i_hsmshape_PSF_e2'] = e_cross_I
        #     star_cat['i_hsmshape_PSF_T'] = T_I

        #     # 4- delta_e_PSF
        #     logger.info('Creating e_PSF and T_PSF residual maps.')
        #     mPSFresstar, delta_e_plus, delta_e_cross, delta_T, e_plus_I, e_cross_I = self.make_PSF_res_maps(star_cat, fsk)
        #     logger.info("Computing w2e2.")
        #     w2e2 = self.get_w2e2(star_cat, delta_e_plus, delta_e_cross, fsk)
        #     # Write e_PSFres map
        #     logger.info("Writing output to {}.".format(self.get_output('ePSFres_map')))
        #     header = fsk.wcs.to_header()
        #     hdus = []
        #     shp_mp = [fsk.ny, fsk.nx]
        #     # Maps
        #     head = header.copy()
        #     head['DESCR'] = ('e_PSFres1', 'Description')
        #     hdu = fits.PrimaryHDU(data=mPSFresstar[0][0].reshape(shp_mp),
        #                               header=head)
        #     hdus.append(hdu)
        #     head = header.copy()
        #     head['DESCR'] = ('e_PSFres2', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFresstar[0][1].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     head = header.copy()
        #     head['DESCR'] = ('e_PSFres weight mask', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFresstar[1][0].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     head['DESCR'] = ('e_PSFres binary mask', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFresstar[1][1].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     head['DESCR'] = ('counts map (PSF star sample)', 'Description')
        #     hdu = fits.ImageHDU(data=mPSFresstar[1][2].reshape(shp_mp),
        #                         header=head)
        #     hdus.append(hdu)
        #     # w2e2
        #     cols = [fits.Column(name='w2e2', array=np.atleast_1d(w2e2), format='E')]
        #     hdus.append(fits.BinTableHDU.from_columns(cols))
        #     hdulist = fits.HDUList(hdus)
        #     hdulist.writeto(self.get_output('ePSFres_map'), overwrite=True)
        #     # Write TPSFres map
        #     fsk.write_flat_map(self.get_output('TPSFres_map'),
        #                        np.array([mPSFresstar[2], mPSFresstar[2].astype('bool').astype('int')]),
        #                        descript=['T_PSFres', 'T_PSFres binary mask'])
        #     star_cat['i_shape_delta_PSF_e1'] = delta_e_plus
        #     star_cat['i_shape_delta_PSF_e2'] = delta_e_cross
        #     star_cat['i_shape_delta_PSF_T'] = delta_T
        #     star_cat['i_hsmshape_e1'] = e_plus_I
        #     star_cat['i_hsmshape_e2'] = e_cross_I
        #     star_cat.write(self.get_output('star_catalog_final'), overwrite=True)
        # else:
        #     logger.info('Star catalog not provided. Not generating e_PSF, e_PSF residual maps.')

        # 5- Binary BO mask
        # mask_bo, fsg = self.make_bo_mask(cat[sel_area], fsk,
        #                                  mask_fulldepth=True)
        # fsg.write_flat_map(self.get_output('bo_mask'), mask_bo,
        #                    descript='Bright-object mask')

        # 6- Masked fraction
        masked_fraction_cont = self.make_masked_fraction(cat, fsk,
                                                         mask_fulldepth=True)
        fsk.write_flat_map(self.get_output('masked_fraction'),
                           masked_fraction_cont,
                           descript='Masked fraction')

        # 7- Compute depth map
        # depth, desc = self.make_depth_map(star_cat, fsk)
        # fsk.write_flat_map(self.get_output('depth_map'),
        #                    depth, descript=desc)

        # seeing, seeing_desc = self.make_seeing_map(star_cat, fsk)
        # fsk.write_flat_map(self.get_output('seeing_map'),
        #                    seeing, descript=seeing_desc)

        # sel_binary_stars = self.get_binarystar_flags(cat)
        # sel_binary_stars = ~sel_binary_stars
        sel = ~(sel_raw*sel_clean*sel_maglim*sel_gals*sel_fluxcut*sel_blended)
        print("final size", )
        logger.info("Will lose %d objects to depth, S/N, FDFC, BO mask, and stars" %
                    (np.sum(sel)))
        cat.remove_rows(sel)
        logger.info('Final catalog size: %d' % (len(cat)))
        print('Final catalog size: %d' % (len(cat)))

        ####
        # Implement final cuts
        # - Mag. limit
        # - S/N cut
        # - Star-galaxy separator
        # - Blending
        # sel = ~(sel_raw*sel_clean*sel_maglim*sel_gals*sel_fluxcut*sel_blended)
        # print("final size", )
        # logger.info("Will lose %d objects to depth, S/N, FDFC, BO mask, and stars" %
        #             (np.sum(sel)))
        # cat.remove_rows(sel)

        ####
        # Define shear catalog
        # cat['shear_cat'] = self.shear_cut(cat)

        ####
        # Photo-z binning
        cat['tomo_bin'] = self.pz_binning(cat)


        ####
        # Add multiplicative bias to shear catalog
        # Get multiplicative bias from data
        hdul1 = fits.open(self.config['clean_catalog_data']) 
        mhat_arr = np.zeros(4)
        for i in range(len(mhat_arr)):
            mhat_arr[i] = hdul1[0].header['MHAT_'+str(i+1)]
        msel_arr = np.zeros(4)
        # Measure multiplicative selection bias from data
        cat_data = hdul1[1].data
        if 'ntomo_bins' in self.config:
            self.bin_indxs = self.config['ntomo_bins']
        else:
            self.bin_indxs = range(self.nbins)
        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat_data['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat_data['tomo_bin'] >= 0)
            subcat = cat_data[msk_bin]
            msel_arr[ibin] = self.get_sel_bias(subcat['i_hsmshaperegauss_derived_weight'], 
                subcat['i_apertureflux_10_mag'], subcat['i_hsmshaperegauss_resolution'])[0]
        # Correction factor to account for finite resolution, shell thickness, n(z) differences between data and mocks
        # Need to update this, current values are from Xiangchong
        corr_arr=np.array([1.17133725, 1.08968149, 1.06929737, 1.05591374])
        cat_out = self.add_mbias(cat, mhat_arr, msel_arr, corr_arr)

        ####
        # Secondary peak cut flag - defined to be 0 if included in peak cut
        # width95_mizuki = cat['mizuki_photoz_err95_max'] - cat['mizuki_photoz_err95_min']
        # width95_dnnz = cat['dnnz_photoz_err95_max'] - cat['dnnz_photoz_err95_min']
        # cat['pz_secondary_peak'] = np.logical_or(np.logical_or(cat['tomo_bin']==2, cat['tomo_bin']==3), np.logical_and(width95_mizuki<2.7, width95_dnnz<2.7))

        ####
        # Calibrated shears
        # e1c, e2c, mhat, resp = self.shear_calibrate(cat)
        # cat['i_hsmshaperegauss_e1_calib'] = e1c
        # cat['i_hsmshaperegauss_e2_calib'] = e2c

        ####
        # Write final catalog
        # 1- header
        logger.info("Writing output")
        hdr = fits.Header()
        # for ibin in range(self.nbins):
        #     hdr['MHAT_%d' % (ibin+1)] = mhat[ibin]
        # for ibin in range(self.nbins):
        #     hdr['RESPONS_%d' % (ibin+1)] = resp[ibin]
        hdr['BAND'] = self.config['band']
        hdr['DEPTH'] = self.config['depth_cut']
        prm_hdu = fits.PrimaryHDU(header=hdr)
        # 2- Catalog
        cat_hdu = fits.table_to_hdu(cat)
        # 3- Actual writing
        hdul = fits.HDUList([prm_hdu, cat_hdu])
        hdul.writeto(self.get_output('clean_catalog'), overwrite=True)
        ####

        ####
        # Plotting
        # for i_d, d in enumerate(dustmaps):
        #     plot_map(self.config, fsk, d, 'dust_%d' % i_d)
        # plot_map(self.config, fsk, mstar, 'Nstar')
        # if self.get_input('star_catalog') != 'NONE':
        #     plot_map(self.config, fsk, mPSFstar[0][0], 'e_PSF1')
        #     plot_map(self.config, fsk, mPSFstar[0][1], 'e_PSF2')
        #     plot_map(self.config, fsk, mPSFstar[1][0], 'e_PSF_w')
        #     plot_map(self.config, fsk, mPSFstar[1][1], 'e_PSF_m')
        #     plot_map(self.config, fsk, mPSFstar[1][2], 'e_PSF_c')
        #     plot_map(self.config, fsk, mPSFresstar[0][0], 'e_PSFres1')
        #     plot_map(self.config, fsk, mPSFresstar[0][1], 'e_PSFres2')
        #     plot_map(self.config, fsk, mPSFresstar[1][0], 'e_PSFres_w')
        #     plot_map(self.config, fsk, mPSFresstar[1][1], 'e_PSFres_m')
        #     plot_map(self.config, fsk, mPSFresstar[1][2], 'e_PSFres_c')
        # # plot_map(self.config, fsg, mask_bo, 'bo_mask')
        # plot_map(self.config, fsk, masked_fraction_cont, 'masked_fraction')
        # plot_map(self.config, fsk, depth, 'depth_map')
        # plot_histo(self.config, 'cmodel_mags',
        #            [cat['%s_cmodel_mag' % b] for b in self.bands],
        #            ['m_%s' % b for b in self.bands], bins=100, logy=True)
        ####

        # Permissions on NERSC
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

if __name__ == '__main__':
    cls = PipelineStage.main()
