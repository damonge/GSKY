from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table, vstack
import numpy as np
from .flatmaps import FlatMapInfo
from .map_utils import createCountsMap, createMeanStdMaps, createMask, removeDisconnected, createSpin2Map
from .estDepth import get_depth
from astropy.io import fits
import copy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReduceCat(PipelineStage):
    name = "ReduceCat"
    inputs = [('raw_data', FitsFile)]
    outputs = [('clean_catalog', FitsFile),
               ('dust_map', FitsFile),
               ('star_map', FitsFile),
               ('bo_mask', FitsFile),
               ('masked_fraction', FitsFile),
               ('depth_map', FitsFile),
               ('ePSF_map', FitsFile),
               ('ePSFres_map', FitsFile)]
    config_options = {'min_snr': 10., 'depth_cut': 24.5,
                      'mapping': {'wcs': None, 'res': 0.0285,
                                  'res_bo': 0.003, 'pad': 0.1,
                                  'projection': 'CAR'},
                      'band': 'i', 'depth_method': 'fluxerr',
                      'shearrot': 'flipqu', 'mask_type': 'sirius',
                      'ra':  'ra', 'dec':  'dec',
                      'pz_code': 'ephor_ab', 'pz_mark': 'best',
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
        mstar = createCountsMap(cat[self.config['ra']][sel],
                                cat[self.config['dec']][sel],
                                fsk)+0.
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
        if mask_fulldepth:
            masked *= cat['wl_fulldepth_fullcolor']
        if self.config['mask_type'] == 'arcturus':
            masked *= cat['mask_Arcturus']
        elif self.config['mask_type'] == 'sirius':
            masked *= np.logical_not(cat['iflags_pixel_bright_object_center'])
            masked *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        else:
            raise ValueError('Mask type '+self.config['mask_type'] +
                             ' not supported')
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
        snrs = cat['%scmodel_flux' % band]/cat['%scmodel_flux_err' % band]
        if method == 'fluxerr':
            arr1 = cat['%scmodel_flux_err' % band]
            arr2 = None
        else:
            arr1 = cat['%scmodel_mag' % band]
            arr2 = snrs
        depth, _ = get_depth(method, cat[self.config['ra']],
                             cat[self.config['dec']],
                             arr1=arr1, arr2=arr2,
                             fsk=fsk, snrthreshold=self.config['min_snr'],
                             interpolate=True, count_threshold=4)
        desc = '%d-s depth, ' % (self.config['min_snr'])+band+' '+method+' mean'

        return depth, desc

    def make_PSF_maps(self, cat, fsk, sel):
        """
        Get e_PSF, 1, e_PSF, 2 maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # PSF of stars
        star_cat = copy.deepcopy(cat)[sel]
        Mxx = star_cat['ishape_hsm_psfmoments_11']
        Myy = star_cat['ishape_hsm_psfmoments_22']
        Mxy = star_cat['ishape_hsm_psfmoments_12']
        T_I = Mxx + Myy
        e_plus_I = (Mxx - Myy)/T_I
        e_cross_I = 2*Mxy/T_I
        ePSFmaps, ePSFmasks = createSpin2Map(star_cat[self.config['ra']],
                                             star_cat[self.config['dec']],
                                             e_plus_I, e_cross_I, fsk,
                                             weights=star_cat['ishape_hsm_regauss_derived_shape_weight'],
                                             shearrot=self.config['shearrot'])

        maps = [ePSFmaps, ePSFmasks]

        return maps

    def make_PSF_res_maps(self, cat, fsk, sel):
        """
        Get e_PSF, 1, e_PSF, 2 residual maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # PSF of stars
        star_cat = copy.deepcopy(cat)[sel]
        Mxx = star_cat['ishape_hsm_psfmoments_11']
        Myy = star_cat['ishape_hsm_psfmoments_22']
        Mxy = star_cat['ishape_hsm_psfmoments_12']
        T_I = Mxx + Myy
        e_plus_PSF = (Mxx - Myy)/T_I
        e_cross_PSF = 2*Mxy/T_I

        Mxx = star_cat['ishape_hsm_moments_11']
        Myy = star_cat['ishape_hsm_moments_22']
        Mxy = star_cat['ishape_hsm_moments_12']
        T_I = Mxx + Myy
        e_plus_I = (Mxx - Myy)/T_I
        e_cross_I = 2*Mxy/T_I

        delta_e_plus = e_plus_PSF - e_plus_I
        delta_e_cross = e_cross_PSF - e_cross_I

        ePSFresmaps, ePSFresmasks = createSpin2Map(star_cat[self.config['ra']],
                                                   star_cat[self.config['dec']],
                                                   delta_e_plus, delta_e_cross, fsk,
                                                   weights=star_cat['ishape_hsm_regauss_derived_shape_weight'],
                                                   shearrot=self.config['shearrot'])

        maps = [ePSFresmaps, ePSFresmasks]

        return maps

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

        shearmask = ishape_flags_mask*ishape_sigma_mask*ishape_resolution_mask*ishape_shear_mod_mask
        return shearmask

    def shear_calibrate(self, cat):
        # Galaxies used for shear
        mask_shear = cat['shear_cat'] & (cat['tomo_bin'] >= 0)

        # Compute responsivity
        resp = 1. - np.average(cat[mask_shear]['ishape_hsm_regauss_derived_rms_e']**2,
                               weights=cat[mask_shear]['ishape_hsm_regauss_derived_shape_weight'])

        # Calibrate shears per redshift bin
        e1cal = np.zeros(len(cat))
        e2cal = np.zeros(len(cat))
        mhats = np.zeros(self.nbins)
        for ibin in range(self.nbins):
            mask_bin = mask_shear & (cat['tomo_bin'] == ibin)
            mhat = np.average(cat[mask_bin]['ishape_hsm_regauss_derived_shear_bias_m'],
                              weights=cat[mask_bin]['ishape_hsm_regauss_derived_shape_weight'])
            mhats[ibin] = mhat
            e1 = (cat[mask_bin]['ishape_hsm_regauss_e1']/(2.*resp) -
                  cat[mask_bin]['ishape_hsm_regauss_derived_shear_bias_c1']) / mhat
            e2 = (cat[mask_bin]['ishape_hsm_regauss_e2']/(2.*resp) -
                  cat[mask_bin]['ishape_hsm_regauss_derived_shear_bias_c2']) / mhat
            e1cal[mask_bin] = e1
            e2cal[mask_bin] = e2
        return e1cal, e2cal, mhats, resp

    def pz_binning(self, cat):
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][1:]
        self.nbins = len(zi_arr)

        if self.config['pz_code'] == 'ephor_ab':
            self.pz_code = 'eab'
        elif self.config['pz_code'] == 'frankenz':
            self.pz_code = 'frz'
        elif self.config['pz_code'] == 'nnpz':
            self.pz_code = 'nnz'
        else:
            raise KeyError("Photo-z method "+self.config['pz_code'] +
                           " unavailable. Choose ephor_ab, frankenz or nnpz")

        if self.config['pz_mark'] not in ['best', 'mean', 'mode', 'mc']:
            raise KeyError("Photo-z mark "+self.config['pz_mark'] +
                           " unavailable. Choose between "
                           "best, mean, mode and mc")

        self.column_mark = 'pz_'+self.config['pz_mark']+'_'+self.pz_code
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
        f = open(self.get_input('raw_data'))
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

        # Collect sample cuts
        sel_area = cat['wl_fulldepth_fullcolor']
        sel_clean = sel_area & cat['clean_photometry']
        sel_maglim = np.ones(len(cat), dtype=bool)
        sel_maglim[cat['%scmodel_mag' % band] -
                   cat['a_%s' % band] > self.config['depth_cut']] = 0
        # Blending
        sel_blended = np.ones(len(cat), dtype=bool)
        # abs_flux<10^-0.375
        sel_blended[cat['iblendedness_abs_flux'] >= 0.42169650342] = 0
        # S/N in i
        sel_fluxcut_i = np.ones(len(cat), dtype=bool)
        sel_fluxcut_i[cat['icmodel_flux'] < 10*cat['icmodel_flux_err']] = 0
        # S/N in g
        sel_fluxcut_g = np.ones(len(cat), dtype=int)
        sel_fluxcut_g[cat['gcmodel_flux'] < 5*cat['gcmodel_flux_err']] = 0
        # S/N in r
        sel_fluxcut_r = np.ones(len(cat), dtype=int)
        sel_fluxcut_r[cat['rcmodel_flux'] < 5*cat['rcmodel_flux_err']] = 0
        # S/N in z
        sel_fluxcut_z = np.ones(len(cat), dtype=int)
        sel_fluxcut_z[cat['zcmodel_flux'] < 5*cat['zcmodel_flux_err']] = 0
        # S/N in y
        sel_fluxcut_y = np.ones(len(cat), dtype=int)
        sel_fluxcut_y[cat['ycmodel_flux'] < 5*cat['ycmodel_flux_err']] = 0
        # S/N in grzy (at least 2 pass)
        sel_fluxcut_grzy = (sel_fluxcut_g+sel_fluxcut_r +
                            sel_fluxcut_z+sel_fluxcut_y >= 2)
        # Overall S/N
        sel_fluxcut = sel_fluxcut_i*sel_fluxcut_grzy
        # Stars
        sel_stars = np.ones(len(cat), dtype=bool)
        sel_stars[cat['iclassification_extendedness'] > 0.99] = 0
        # Galaxies
        sel_gals = np.ones(len(cat), dtype=bool)
        sel_gals[cat['iclassification_extendedness'] < 0.99] = 0

        ####
        # Generate sky projection
        fsk = FlatMapInfo.from_coords(cat[sel_area][self.config['ra']],
                                      cat[sel_area][self.config['dec']],
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
        mstar, descstar = self.make_star_map(cat, fsk,
                                             sel_clean *
                                             sel_maglim *
                                             sel_stars *
                                             sel_fluxcut *
                                             sel_blended)
        fsk.write_flat_map(self.get_output('star_map'), mstar,
                           descript=descstar)

        # 3- e_PSF
        # TODO: do these stars need to have the same cuts as our sample?
        logger.info('Creating e_PSF map.')
        mPSFstar = self.make_PSF_maps(cat, fsk,
                                      sel_clean *
                                      sel_maglim *
                                      sel_stars *
                                      sel_fluxcut *
                                      sel_blended)
        fsk.write_flat_map(self.get_output('ePSF_map'),
                           np.array([mPSFstar[0][0], mPSFstar[0][1],
                                     mPSFstar[1][0], mPSFstar[1][1],
                                     mPSFstar[1][2]]),
                           descript=['e_PSF1', 'e_PSF2',
                                     'e_PSF weight mask', 'e_PSF binary mask',
                                     'counts map (PSF star sample)'])

        # 4- delta_e_PSF
        logger.info('Creating e_PSF residual map.')
        mPSFresstar = self.make_PSF_res_maps(cat, fsk,
                                             sel_clean *
                                             sel_maglim *
                                             sel_stars *
                                             sel_fluxcut *
                                             sel_blended)
        fsk.write_flat_map(self.get_output('ePSFres_map'),
                           np.array([mPSFresstar[0][0], mPSFresstar[0][1],
                                     mPSFresstar[1][0], mPSFresstar[1][1],
                                     mPSFresstar[1][2]]),
                           descript=['e_PSFres1', 'e_PSFres2',
                                     'e_PSFres weight mask',
                                     'e_PSFres binary mask',
                                     'counts map (PSF star sample)'])

        # 5- Binary BO mask
        mask_bo, fsg = self.make_bo_mask(cat[sel_area], fsk,
                                         mask_fulldepth=True)
        fsg.write_flat_map(self.get_output('bo_mask'), mask_bo,
                           descript='Bright-object mask')

        # 6- Masked fraction
        masked_fraction_cont = self.make_masked_fraction(cat, fsk,
                                                         mask_fulldepth=True)
        fsk.write_flat_map(self.get_output('masked_fraction'),
                           masked_fraction_cont,
                           descript='Masked fraction')

        # 7- Compute depth map
        depth, desc = self.make_depth_map(cat[sel_stars], fsk)
        fsk.write_flat_map(self.get_output('depth_map'),
                           depth, descript=desc)

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
        cat['shear_cat'] = self.shear_cut(cat)

        ####
        # Photo-z binning
        cat['tomo_bin'] = self.pz_binning(cat)

        ####
        # Calibrated shears
        e1c, e2c, mhat, resp = self.shear_calibrate(cat)
        cat['ishape_hsm_regauss_e1_calib'] = e1c
        cat['ishape_hsm_regauss_e2_calib'] = e2c

        ####
        # Write final catalog
        # 1- header
        logger.info("Writing output")
        hdr = fits.Header()
        for ibin in range(self.nbins):
            hdr['MHAT_%d' % (ibin+1)] = mhat[ibin]
        hdr['RESPONS'] = resp
        hdr['BAND'] = self.config['band']
        hdr['DEPTH'] = self.config['depth_cut']
        prm_hdu = fits.PrimaryHDU(header=hdr)
        # 2- Catalog
        cat_hdu = fits.table_to_hdu(cat)
        # 3- Actual writing
        hdul = fits.HDUList([prm_hdu, cat_hdu])
        hdul.writeto(self.get_output('clean_catalog'), overwrite=True)
        ####


if __name__ == '__main__':
    cls = PipelineStage.main()
