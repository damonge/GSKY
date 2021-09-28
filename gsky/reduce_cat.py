from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os
import healpy as hp
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


class ReduceCat(PipelineStage):
    name = "ReduceCat"
    inputs = [('cut_map', FitsFile),
              ('raw_data', FitsFile),
              ('star_catalog', FitsFile)]
    # outputs = [('clean_catalog', FitsFile),
    #            ('dust_map', FitsFile),
    #            ('star_map', FitsFile),
    #            ('bo_mask', FitsFile),
    #            ('masked_fraction', FitsFile),
    #            ('depth_map', FitsFile),
    #            ('ePSF_map', FitsFile),
    #            ('ePSFres_map', FitsFile),
    #            ('TPSF_map', FitsFile),
    #            ('TPSFres_map', FitsFile),
    #            ('star_catalog_matched', FitsFile)]
    outputs = [('clean_catalog', FitsFile),
               ('dust_map', FitsFile),
               ('star_map', FitsFile),
               ('masked_fraction', FitsFile),
               ('depth_map', FitsFile),
               ('seeing_map', FitsFile)]
    config_options = {'plots_dir': None,
                      'min_snr': 10., 'depth_cut': 24.5,
                      'mapping': {'wcs': None, 'res': 0.0285,
                                  'res_bo': 0.003, 'pad': 0.1,
                                  'projection': 'CAR'},
                      'band': 'i', 'depth_method': 'fluxerr',
                      'shearrot': 'noflip', 'mask_type': 'sirius',
                      'ra':  'i_ra', 'dec':  'i_dec',
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
        masked = np.ones(len(cat))
        # full depth full color cut based on healpix map
        hpfname =   "/tigress/rdalal/s19a_shear/s19a_fdfc_hp_contarea_izy-gt-5_trimmed_fd001.fits"
        m       =   hp.read_map(hpfname, nest = True, dtype = np.bool)
        mfactor =   np.pi/180.
        indices_map =   np.where(m)[0]
        nside   =   hp.get_nside(m)
        phi     =   cat[self.config['ra']]*mfactor
        theta   =   np.pi/2. - cat[self.config['dec']]*mfactor
        indices_obj = hp.ang2pix(nside, theta, phi, nest = True)
        masked *= np.in1d(indices_obj, indices_map)

        # bright object mask
        masked *= np.logical_not(cat['i_mask_brightstar_ghost'])
        masked *= np.logical_not(cat['i_mask_brightstar_halo'])
        masked *= np.logical_not(cat['i_mask_brightstar_blooming'])
        #masked2 = masked*np.logical_not(cat['weak_lensing_flag'])
        #print("Raw data count", np.sum(masked2))
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
        logger.info("Creating depth maps")
        #method = self.config['depth_method']
        band = self.config['band']
        # good_object_id = np.ones(len(cat))
        # good_object_id *= np.logical_not(cat['i_mask_brightstar_ghost'])
        psf_11 = cat['i_sdssshape_shape11'][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22']))]
        psf_22 = cat['i_sdssshape_shape22'][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22']))]
        print("psf_11", np.min(psf_11), np.max(psf_11))
        print(np.sum(np.isnan(psf_11)))
        print("psf_22", np.min(psf_22), np.max(psf_22))
        print(np.sum(np.isnan(psf_22)))
        arr1 = np.sqrt(0.5*(psf_11+psf_22))
        print(np.min(arr1), np.max(arr1))
        print("Mean seeing", np.mean(arr1))
        seeing, _ = get_seeing(cat[self.config['ra']][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22'])) & np.logical_not(arr1>5.0)],
                             cat[self.config['dec']][np.logical_not(np.isnan(cat['i_sdssshape_shape11'])) & np.logical_not(np.isnan(cat['i_sdssshape_shape22'])) & np.logical_not(arr1>5.0)],
                             arr1=arr1[np.logical_not(arr1>5.0)],
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
        Mxx = star_cat['ishape_sdss_psf_11']
        Myy = star_cat['ishape_sdss_psf_22']
        Mxy = star_cat['ishape_sdss_psf_12']
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
        Mxx = star_cat['ishape_sdss_psf_11']
        Myy = star_cat['ishape_sdss_psf_22']
        Mxy = star_cat['ishape_sdss_psf_12']
        T_PSF = Mxx + Myy
        e_plus_PSF = (Mxx - Myy)/T_PSF
        e_cross_PSF = 2*Mxy/T_PSF

        Mxx = star_cat['ishape_sdss_11']
        Myy = star_cat['ishape_sdss_22']
        Mxy = star_cat['ishape_sdss_12']
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

        ishape_flags_mask = ~cat['ishape_hsm_regauss_flags']
        ishape_sigma_mask = ~np.isnan(cat['ishape_hsm_regauss_sigma'])
        ishape_resolution_mask = cat['ishape_hsm_regauss_resolution'] >= 0.3
        ishape_shear_mod_mask = (cat['ishape_hsm_regauss_e1']**2 + cat['ishape_hsm_regauss_e2']**2) < 2
        ishape_sigma_mask *= (cat['ishape_hsm_regauss_sigma'] >= 0.)*(cat['ishape_hsm_regauss_sigma'] <= 0.4)
        # Remove masked objects
        if self.config['mask_type'] == 'arcturus':
            star_mask = cat['mask_Arcturus'].astype(bool)
        elif self.config['mask_type'] == 'sirius':
            star_mask = np.logical_not(cat['iflags_pixel_bright_object_center'])
            star_mask *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        else:
            raise KeyError("Mask type "+self.config['mask_type'] +
                           " not supported. Choose arcturus or sirius")
        fdfc_mask = cat['wl_fulldepth_fullcolor']

        shearmask = ishape_flags_mask*ishape_sigma_mask*ishape_resolution_mask*ishape_shear_mod_mask*star_mask*fdfc_mask
        
        return shearmask

    def shear_calibrate(self, cat):
        # Galaxies used for shear
        mask_shear = cat['shear_cat'] & (cat['tomo_bin'] >= 0)

        # Calibrate shears per redshift bin
        e1cal = np.zeros(len(cat))
        e2cal = np.zeros(len(cat))
        mhats = np.zeros(self.nbins)
        resps = np.zeros(self.nbins)
        for ibin in range(self.nbins):
            mask_bin = mask_shear & (cat['tomo_bin'] == ibin)
            # Compute multiplicative bias
            mhat = np.average(cat[mask_bin]['ishape_hsm_regauss_derived_shear_bias_m'],
                              weights=cat[mask_bin]['ishape_hsm_regauss_derived_shape_weight'])
            mhats[ibin] = mhat
            # Compute responsivity
            resp = 1. - np.average(cat[mask_bin]['ishape_hsm_regauss_derived_rms_e'] ** 2,
                                   weights=cat[mask_bin]['ishape_hsm_regauss_derived_shape_weight'])
            resps[ibin] = resp

            e1 = (cat[mask_bin]['ishape_hsm_regauss_e1']/(2.*resp) -
                  cat[mask_bin]['ishape_hsm_regauss_derived_shear_bias_c1']) / (1 + mhat)
            e2 = (cat[mask_bin]['ishape_hsm_regauss_e2']/(2.*resp) -
                  cat[mask_bin]['ishape_hsm_regauss_derived_shear_bias_c2']) / (1 + mhat)
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

        w2e2maps = createW2QU2Map(cat['ra'],
                                  cat['dec'],
                                  e1,
                                  e2, fsk,
                                  weights=None)

        w2e2 = 0.5*(np.mean(w2e2maps[0]) + np.mean(w2e2maps[1]))

        return w2e2

    def pz_binning(self, cat):
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][1:]
        self.nbins = len(zi_arr)

        if self.config['pz_code'] == 'dnnz':
            self.pz_code = 'dnnz'
        # elif self.config['pz_code'] == 'frankenz':
        #     self.pz_code = 'frz'
        # elif self.config['pz_code'] == 'nnpz':
        #     self.pz_code = 'nnz'
        else:
            raise KeyError("Photo-z method "+self.config['pz_code'] +
                           " unavailable. Choose dnnz")

        if self.config['pz_mark'] not in ['best', 'mean', 'mode', 'mc']:
            raise KeyError("Photo-z mark "+self.config['pz_mark'] +
                           " unavailable. Choose between "
                           "best, mean, mode and mc")

        # self.column_mark = 'pz_'+self.config['pz_mark']+'_'+self.pz_code
        self.column_mark = self.pz_code+'_'+'photoz_'+self.config['pz_mark']
        zs = cat[self.column_mark]

        # Assign all galaxies to bin -1
        bin_number = np.zeros(len(cat), dtype=int) - 1

        for ib, (zi, zf) in enumerate(zip(zi_arr, zf_arr)):
            msk = (zs <= zf) & (zs > zi)
            bin_number[msk] = ib
        return bin_number

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

        # Read list of files
        f = open(self.get_input('cut_map'))
        files = [s.strip() for s in f.readlines()]
        f.close()

        # Read catalog
        cat = Table.read(files[0])
        if len(cat) > 1:
            for fname in files[1:]:
                c = Table.read(fname)
                cat = vstack([cat, c], join_type='exact')

        if band not in self.bands:
            raise ValueError("Band "+band+" not available")

        logger.info('Initial catalog size: %d' % (len(cat)))

        # Clean nulls and nans
        logger.info("Basic cleanup")
        sel = np.ones(len(cat), dtype=bool)
        isnull_names = []
        for key in cat.keys():
            if key.__contains__('isnull'):
                if not key.startswith('ishape'):
                    sel[cat[key]] = 0
                isnull_names.append(key)
            else:
                # Keep photo-zs and shapes even if they're NaNs
                if (not key.startswith("pz_")) and (not key.startswith('ishape')):
                    sel[np.isnan(cat[key])] = 0
        logger.info("Will drop %d rows" % (len(sel)-np.sum(sel)))
        cat.remove_columns(isnull_names)
        cat.remove_rows(~sel)

        # Read raw catalog, for getting masked fraction
        cat_raw = Table.read(self.get_input('raw_data'))

        # Collect sample cuts
        #sel_area = cat['wl_fulldepth_fullcolor']
        # sel_clean = cat['clean_photometry']
        sel_area = np.ones(len(cat), dtype=bool)
        sel_clean = np.ones(len(cat), dtype=bool)
        sel_maglim = np.ones(len(cat), dtype=bool)
        # sel_maglim[cat['%sc_model_mag' % band] -
        #            cat['a_%s' % band] > self.config['depth_cut']] = 0
        # Blending
        sel_blended = np.ones(len(cat), dtype=bool)
        # abs_flux<10^-0.375
        # sel_blended[cat['iblendedness_abs_flux'] >= 0.42169650342] = 0
        # S/N in i
        sel_fluxcut_i = np.ones(len(cat), dtype=bool)
        # sel_fluxcut_i[cat['i_cmodel_flux'] < 10*cat['i_cmodel_flux_err']] = 0
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
        # Generate sky projection
        fsk = FlatMapInfo.from_coords(cat[self.config['ra']],
                                      cat[self.config['dec']],
                                      self.mpp)

        ####
        # Generate systematics maps
        # 1- Dust
        dustmaps, dustdesc = self.make_dust_map(cat, fsk)
        fsk.write_flat_map(self.get_output('dust_map'), np.array(dustmaps),
                           descript=dustdesc)

        # 2- Nstar
        #    This needs to be done for stars passing the same cuts as the
        #    sample (except for the s/g separator)
        # Above magnitude limit

        if self.get_input('star_catalog') != 'NONE':
            logger.info('Reading star catalog from {}.'.format(self.get_input('star_catalog')))
            hdul = fits.open(self.get_input('star_catalog'))
            star_cat = hdul[1].data

        mstar, descstar = self.make_star_map(star_cat, fsk,
                                             sel_clean *
                                             sel_maglim *
                                             sel_stars *
                                             sel_fluxcut *
                                             sel_blended)
        fsk.write_flat_map(self.get_output('star_map'), mstar,
                           descript=descstar)

        # 3- e_PSF
        # if self.get_input('star_catalog') != 'NONE':
        #     logger.info('Reading star catalog from {}.'.format(self.get_input('star_catalog')))
        #     hdul = fits.open(self.get_input('star_catalog'))
        #     star_cat = hdul[1].data
            # TODO: do these stars need to have the same cuts as our sample?
            # star_cat_matched = self.match_star_cats(cat, sel_clean*sel_psf_valid*sel_stars, star_cat)
            # logger.info('Creating e_PSF and T_PSF maps.')
            # mPSFstar, e_plus_I, e_cross_I, T_I = self.make_PSF_maps(star_cat_matched, fsk)
            # logger.info("Computing w2e2.")
            # w2e2 = self.get_w2e2(star_cat_matched, e_plus_I, e_cross_I, fsk)
            # logger.info("Writing output to {}.".format(self.get_output('ePSF_map')))
            # header = fsk.wcs.to_header()
            # hdus = []
            # shp_mp = [fsk.ny, fsk.nx]
            # Maps
            # head = header.copy()
            # head['DESCR'] = ('e_PSF1', 'Description')
            # hdu = fits.PrimaryHDU(data=mPSFstar[0][0].reshape(shp_mp),
            #                           header=head)
            # hdus.append(hdu)
            # head = header.copy()
            # head['DESCR'] = ('e_PSF2', 'Description')
            # hdu = fits.ImageHDU(data=mPSFstar[0][1].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # head = header.copy()
            # head['DESCR'] = ('e_PSF weight mask', 'Description')
            # hdu = fits.ImageHDU(data=mPSFstar[1][0].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # head['DESCR'] = ('e_PSF binary mask', 'Description')
            # hdu = fits.ImageHDU(data=mPSFstar[1][1].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # head['DESCR'] = ('counts map (PSF star sample)', 'Description')
            # hdu = fits.ImageHDU(data=mPSFstar[1][2].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # # w2e2
            # cols = [fits.Column(name='w2e2', array=np.atleast_1d(w2e2), format='E')]
            # hdus.append(fits.BinTableHDU.from_columns(cols))
            # hdulist = fits.HDUList(hdus)
            # hdulist.writeto(self.get_output('ePSF_map'), overwrite=True)

            # fsk.write_flat_map(self.get_output('TPSF_map'),
            #                    np.array([mPSFstar[2], mPSFstar[2].astype('bool').astype('int')]),
            #                    descript=['T_PSF', 'T_PSF binary mask'])
            # star_cat_matched['ishape_hsm_PSF_e1'] = e_plus_I
            # star_cat_matched['ishape_hsm_PSF_e2'] = e_cross_I
            # star_cat_matched['ishape_hsm_PSF_T'] = T_I

            # # 4- delta_e_PSF
            # logger.info('Creating e_PSF and T_PSF residual maps.')
            # mPSFresstar, delta_e_plus, delta_e_cross, delta_T, e_plus_I, e_cross_I = self.make_PSF_res_maps(star_cat_matched, fsk)
            # logger.info("Computing w2e2.")
            # w2e2 = self.get_w2e2(star_cat_matched, delta_e_plus, delta_e_cross, fsk)
            # # Write e_PSFres map
            # logger.info("Writing output to {}.".format(self.get_output('ePSFres_map')))
            # header = fsk.wcs.to_header()
            # hdus = []
            # shp_mp = [fsk.ny, fsk.nx]
            # # Maps
            # head = header.copy()
            # head['DESCR'] = ('e_PSFres1', 'Description')
            # hdu = fits.PrimaryHDU(data=mPSFresstar[0][0].reshape(shp_mp),
            #                           header=head)
            # hdus.append(hdu)
            # head = header.copy()
            # head['DESCR'] = ('e_PSFres2', 'Description')
            # hdu = fits.ImageHDU(data=mPSFresstar[0][1].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # head = header.copy()
            # head['DESCR'] = ('e_PSFres weight mask', 'Description')
            # hdu = fits.ImageHDU(data=mPSFresstar[1][0].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # head['DESCR'] = ('e_PSFres binary mask', 'Description')
            # hdu = fits.ImageHDU(data=mPSFresstar[1][1].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # head['DESCR'] = ('counts map (PSF star sample)', 'Description')
            # hdu = fits.ImageHDU(data=mPSFresstar[1][2].reshape(shp_mp),
            #                     header=head)
            # hdus.append(hdu)
            # # w2e2
            # cols = [fits.Column(name='w2e2', array=np.atleast_1d(w2e2), format='E')]
            # hdus.append(fits.BinTableHDU.from_columns(cols))
            # hdulist = fits.HDUList(hdus)
            # hdulist.writeto(self.get_output('ePSFres_map'), overwrite=True)
            # # Write TPSFres map
            # fsk.write_flat_map(self.get_output('TPSFres_map'),
            #                    np.array([mPSFresstar[2], mPSFresstar[2].astype('bool').astype('int')]),
            #                    descript=['T_PSFres', 'T_PSFres binary mask'])
            # star_cat_matched['ishape_delta_PSF_e1'] = delta_e_plus
            # star_cat_matched['ishape_delta_PSF_e2'] = delta_e_cross
            # star_cat_matched['ishape_delta_PSF_T'] = delta_T
            # star_cat_matched['ishape_hsm_e1'] = e_plus_I
            # star_cat_matched['ishape_hsm_e2'] = e_cross_I
            # star_cat_matched.write(self.get_output('star_catalog_matched'), overwrite=True)

        # else:
        #     logger.info('Star catalog not provided. Not generating e_PSF, e_PSF residual maps.')

        # 5- Binary BO mask
        # mask_bo, fsg = self.make_bo_mask(cat[sel_area], fsk,
        #                                  mask_fulldepth=True)
        # fsg.write_flat_map(self.get_output('bo_mask'), mask_bo,
        #                    descript='Bright-object mask')

        # 6- Masked fraction
        masked_fraction_cont = self.make_masked_fraction(cat_raw, fsk,
                                                         mask_fulldepth=True)
        fsk.write_flat_map(self.get_output('masked_fraction'),
                           masked_fraction_cont,
                           descript='Masked fraction')

        # 7- Compute depth map
        depth, desc = self.make_depth_map(star_cat, fsk)
        fsk.write_flat_map(self.get_output('depth_map'),
                           depth, descript=desc)

        seeing, seeing_desc = self.make_seeing_map(star_cat, fsk)
        fsk.write_flat_map(self.get_output('seeing_map'),
                           seeing, descript=seeing_desc)

        ####
        # Implement final cuts
        # - Mag. limit
        # - S/N cut
        # - Star-galaxy separator
        # - Blending
        sel = ~(sel_clean*sel_maglim*sel_gals*sel_fluxcut*sel_blended)
        logger.info("Will lose %d objects to depth, S/N and stars" %
                    (np.sum(sel)))
        cat.remove_rows(sel)

        ####
        # Define shear catalog
        # cat['shear_cat'] = self.shear_cut(cat)

        ####
        # Photo-z binning
        cat['tomo_bin'] = self.pz_binning(cat)

        ####
        # Calibrated shears
        # e1c, e2c, mhat, resp = self.shear_calibrate(cat)
        # cat['ishape_hsm_regauss_e1_calib'] = e1c
        # cat['ishape_hsm_regauss_e2_calib'] = e2c

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
        for i_d, d in enumerate(dustmaps):
            plot_map(self.config, fsk, d, 'dust_%d' % i_d)
        plot_map(self.config, fsk, mstar, 'Nstar')
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
        # plot_map(self.config, fsg, mask_bo, 'bo_mask')
        plot_map(self.config, fsk, masked_fraction_cont, 'masked_fraction')
        plot_map(self.config, fsk, depth, 'depth_map')
        # plot_histo(self.config, 'cmodel_mags',
        #            [cat['%s_cmodel_mag' % b] for b in self.bands],
        #            ['m_%s' % b for b in self.bands], bins=100, logy=True)
        ####

        # Permissions on NERSC
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

if __name__ == '__main__':
    cls = PipelineStage.main()
