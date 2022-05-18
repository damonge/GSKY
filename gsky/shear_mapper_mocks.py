from ceci import PipelineStage
from .types import FitsFile, ASCIIFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createSpin2Map, createW2QU2Map
from astropy.io import fits
import os
from .plot_utils import plot_map, plot_curves

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShearMapperMocks(PipelineStage):
    name = "ShearMapperMocks"
    inputs = [('clean_catalog', FitsFile),
              ('masked_fraction', FitsFile)]
              # ('cosmos_weights', FitsFile),
              # ('pdf_matched', ASCIIFile)]
    outputs = [('gamma_maps', FitsFile),
               ('w2e2_maps', FitsFile)]
    config_options = {'mask_type': 'sirius',
                      'pz_code': 'dnnz',
                      'pz_mark': 'best',
                      'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.50],
                      'nz_bin_num': 200,
                      'nz_bin_max': 3.0,
                      'shearrot': 'noflip',
                      'ra':  'ra_mock', 'dec':  'dec_mock'}

    def get_gamma_maps(self, cat):
        """
        Get gamma1, gamma2 maps and corresponding mask from catalog.
        :param cat:
        :return:
        """

        # if 'i_hsmshaperegauss_e1_calib' not in cat.dtype.names:
        #     raise RuntimeError('get_gamma_maps must be called with '
        #                        'calibrated shear catalog. Aborting.')
        maps = []

        # Tomographic maps
        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat['tomo_bin'] >= 0) 
            subcat = cat[msk_bin]
            gammamaps, gammamasks = createSpin2Map(subcat[self.config['ra']],
                                                   subcat[self.config['dec']],
                                                   subcat['shear1_sim']/(1-subcat['kappa']),
                                                   subcat['shear2_sim']/(1-subcat['kappa']), self.fsk,
                                                   weights=subcat['weight'],
                                                   shearrot=self.config['shearrot'])
            maps_combined = [gammamaps, gammamasks]
            maps.append(maps_combined)

        return maps

    def get_e2rms(self, cat):
        """
        Get e1_2rms, e2_2rms from catalog.
        :param cat:
        :return:
        """

        # if 'i_hsmshaperegauss_e1_calib' not in cat.dtype.names:
        #     raise RuntimeError('get_e2rms must be called with '
        #                        'calibrated shear catalog. Aborting.')
        e2rms_arr = []

        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat['tomo_bin'] >= 0)
            subcat = cat[msk_bin]
            e1_2rms = np.average((subcat['shear1_sim']/(1-subcat['kappa']))**2,
                                 weights=subcat['weight'])
            e2_2rms = np.average((subcat['shear2_sim']/(1-subcat['kappa']))**2,
                                 weights=subcat['weight'])

            e2rms_combined = np.array([e1_2rms, e2_2rms])
            e2rms_arr.append(e2rms_combined)

        return np.array(e2rms_arr)

    def get_w2e2(self, cat, return_maps=False):
        """
        Compute the weighted mean squared ellipticity in a pixel, averaged over the whole map (used for analytic shape
        noise estimation).
        :param cat:
        :return:
        """

        # if 'i_hsmshaperegauss_e1_calib' not in cat.dtype.names:
        #     raise RuntimeError('get_gamma_maps must be called with '
        #                        'calibrated shear catalog. Aborting.')
        w2e2 = []
        w2e2maps = []

        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat['tomo_bin'] >= 0)
            subcat = cat[msk_bin]
            w2e2maps_curr = createW2QU2Map(subcat[self.config['ra']],
                                                   subcat[self.config['dec']],
                                                   subcat['shear1_sim']/(1-subcat['kappa']),
                                                   subcat['shear2_sim']/(1-subcat['kappa']), self.fsk,
                                                   weights=subcat['weight'])

            w2e2_curr = 0.5*(np.mean(w2e2maps_curr[0]) + np.mean(w2e2maps_curr[1]))
            w2e2.append(w2e2_curr)
            w2e2maps.append(w2e2maps_curr)

        if not return_maps:
            return np.array(w2e2)
        else:
            return np.array(w2e2), w2e2maps

    def get_nz_cosmos(self):
        """
        Get N(z) from weighted COSMOS-30band data
        """
        # TODO: this is wrong.
        # We need to include lensing weights when available.
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][1:]

        if self.config['pz_code'] == 'dnnz':
            pz_code = 'dnnz'
        # elif self.config['pz_code'] == 'frankenz':
        #     pz_code = 'frz'
        # elif self.config['pz_code'] == 'nnpz':
        #     pz_code = 'nnz'
        else:
            raise KeyError("Photo-z method "+self.config['pz_code'] +
                           " unavailable. Choose dnnz")

        if self.config['pz_mark'] not in ['best', 'mean', 'mode', 'mc']:
            raise KeyError("Photo-z mark "+self.config['pz_mark'] +
                           " unavailable. Choose between "
                           "best, mean, mode and mc")

        self.column_mark = 'pz_'+self.config['pz_mark']+'_'+pz_code

        weights_file = fits.open(self.get_input('cosmos_weights'))[1].data

        pzs = []
        for zi, zf in zip(zi_arr, zf_arr):
            msk_cosmos = ((weights_file[self.column_mark] <= zf) &
                          (weights_file[self.column_mark] > zi))
            hz, bz = np.histogram(weights_file[msk_cosmos]['PHOTOZ'],
                                  bins=self.config['nz_bin_num'],
                                  range=[0., self.config['nz_bin_max']],
                                  weights=weights_file[msk_cosmos]['weight'])
            hnz, bnz = np.histogram(weights_file[msk_cosmos]['PHOTOZ'],
                                    bins=self.config['nz_bin_num'],
                                    range=[0., self.config['nz_bin_max']])
            ehz = np.zeros(len(hnz))
            ehz[hnz > 0] = (hz[hnz > 0]+0.)/np.sqrt(hnz[hnz > 0]+0.)
            pzs.append([bz[:-1], bz[1:], (hz+0.)/np.sum(hz+0.), ehz])
        return np.array(pzs)

    def get_nz_stack(self, cat, codename):
        """
        Get N(z) from pdf stacks.
        :param cat: object catalog
        :param codename: photoz code name (demp, ephor,
            ephor_ab, frankenz or nnpz).
        """
        from scipy.interpolate import interp1d

        f = fits.open(self.pdf_files[codename])
        p = f[1].data['pdf'][self.msk]
        z = f[2].data['bins']
        sumpdf = np.sum(p, axis=1)
        pdfgood = sumpdf > 0

        weights = cat['ishape_hsm_regauss_derived_shape_weight']
        z_all = np.linspace(0., self.config['nz_bin_max'],
                            self.config['nz_bin_num'] + 1)
        z0 = z_all[:-1]
        z1 = z_all[1:]
        zm = 0.5*(z0+z1)
        pzs = []
        for i in self.bin_indxs:
            if i != -1:
                msk_good = ((cat['tomo_bin'] == i) &
                            pdfgood &
                            cat['shear_cat'])
            else:
                msk_good = ((cat['tomo_bin'] >= 0) &
                            pdfgood &
                            cat['shear_cat'])
            hz_orig = np.sum(weights[msk_good][:, None] * p[msk_good],
                             axis=0)
            hz_orig /= np.sum(hz_orig)
            hzf = interp1d(z, hz_orig, bounds_error=False,
                           fill_value=0.)
            hzm = hzf(zm)

            pzs.append([z0, z1, hzm / np.sum(hzm)])
        f.close()
        return np.array(pzs)

    def run(self):
        """
        Main routine. This stage:
        - Creates gamma1, gamma2 maps and corresponding masks from
          the reduced catalog for a set of redshift bins.
        - Stores the above into a single FITS file.
        """
        logger.info("Reading masked fraction from {}.".format(self.get_input("masked_fraction")))
        self.fsk, _ = read_flat_map(self.get_input("masked_fraction"))
        self.nbins = len(self.config['pz_bins'])-1
        if 'ntomo_bins' in self.config:
            self.bin_indxs = self.config['ntomo_bins']
        else:
            self.bin_indxs = range(self.nbins)

        logger.info("Reading calibrated shear catalog from {}.".format(self.get_input('clean_catalog')))
        hdul = fits.open(self.get_input('clean_catalog'))
        head_cat = hdul[0].header
        # mhats = np.array([head_cat['MHAT_%d' % (ibin+1)]
        #                   for ibin in range(self.nbins)])
        # resps = np.array([head_cat['RESPONS_%d' % (ibin+1)]
        #                   for ibin in range(self.nbins)])
        cat = hdul[1].data
        # Remove masked objects
        # if self.config['mask_type'] == 'arcturus':
        #     self.msk = cat['mask_Arcturus'].astype(bool)
        # elif self.config['mask_type'] == 'sirius':
        #     self.msk = np.logical_not(cat['iflags_pixel_bright_object_center'])
        #     self.msk *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        # else:
        #     raise KeyError("Mask type "+self.config['mask_type'] +
        #                    " not supported. Choose arcturus or sirius")
        # self.msk *= cat['wl_fulldepth_fullcolor']
        # cat = cat[self.msk]

        # logger.info("Reading pdf filenames")
        # data_syst = np.genfromtxt(self.get_input('pdf_matched'),
        #                           dtype=[('pzname', '|U8'),
        #                                  ('fname', '|U256')])
        # self.pdf_files = {n: fn
        #                   for n, fn in zip(np.atleast_1d(data_syst['pzname']),
        #                                    np.atleast_1d(data_syst['fname']))}

        # logger.info("Getting COSMOS N(z)s")
        # pzs_cosmos = self.get_nz_cosmos()

        # logger.info("Getting pdf stacks")
        # pzs_stack = {}
        # for n in self.pdf_files.keys():
        #     pzs_stack[n] = self.get_nz_stack(cat, n)

        logger.info("Computing e2rms.")
        e2rms = self.get_e2rms(cat)

        logger.info("Computing w2e2.")
        if self.get_output('w2e2_maps') != 'NONE':
            logger.info('Saving w2e2 maps.')
            logger.info("Writing output to {}.".format(self.get_output('w2e2_maps')))
            w2e2, w2e2maps = self.get_w2e2(cat, return_maps=True)
        else:
            logger.info('Not saving w2e2 maps.')
            w2e2 = self.get_w2e2(cat)

        logger.info("Creating shear maps and corresponding masks.")
        gammamaps = self.get_gamma_maps(cat)

        logger.info("Writing output to {}.".format(self.get_output('gamma_maps')))
        header = self.fsk.wcs.to_header()
        hdus = []
        shp_mp = [self.fsk.ny, self.fsk.nx]
        for im, m_list in enumerate(gammamaps):
            if im == len(gammamaps) - 1:
                bin_tag = 'all'
            else:
                bin_tag = im + 1

            # Maps
            head = header.copy()
            head['DESCR'] = ('gamma1, bin {}'.format(bin_tag),
                             'Description')
            if im == 0:
                hdu = fits.PrimaryHDU(data=m_list[0][0].reshape(shp_mp),
                                      header=head)
            else:
                hdu = fits.ImageHDU(data=m_list[0][0].reshape(shp_mp),
                                    header=head)

            hdus.append(hdu)
            head = header.copy()
            head['DESCR'] = ('gamma2, bin {}'.format(bin_tag), 'Description')
            hdu = fits.ImageHDU(data=m_list[0][1].reshape(shp_mp),
                                header=head)
            hdus.append(hdu)
            head = header.copy()
            head['DESCR'] = ('gamma weight mask, bin {}'.format(bin_tag),
                             'Description')
            hdu = fits.ImageHDU(data=m_list[1][0].reshape(shp_mp),
                                header=head)
            hdus.append(hdu)
            head['DESCR'] = ('gamma binary mask, bin {}'.format(bin_tag),
                             'Description')
            hdu = fits.ImageHDU(data=m_list[1][1].reshape(shp_mp),
                                header=head)
            hdus.append(hdu)
            head['DESCR'] = ('counts map (shear sample), bin {}'.format(bin_tag),
                             'Description')
            hdu = fits.ImageHDU(data=m_list[1][2].reshape(shp_mp),
                                header=head)
            hdus.append(hdu)

            # cols = [fits.Column(name='z_i', array=pzs_cosmos[im, 0, :],
            #                     format='E'),
            #         fits.Column(name='z_f', array=pzs_cosmos[im, 1, :],
            #                     format='E'),
            #         fits.Column(name='nz_cosmos', array=pzs_cosmos[im, 2, :],
            #                     format='E'),
            #         fits.Column(name='enz_cosmos', array=pzs_cosmos[im, 3, :],
            #                     format='E')]
            # for n in self.pdf_files.keys():
            #     cols.append(fits.Column(name='nz_'+n,
            #                             array=pzs_stack[n][im, 2, :],
            #                             format='E'))
            # hdus.append(fits.BinTableHDU.from_columns(cols))
        # e2rms
        cols = [fits.Column(name='e2rms', array=e2rms, format='2E'),
                fits.Column(name='w2e2', array=w2e2, format='E')]
                # fits.Column(name='mhats', array=mhats, format='E'),
                # fits.Column(name='resps', array=resps, format='E')]
        hdus.append(fits.BinTableHDU.from_columns(cols))

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.get_output('gamma_maps'), overwrite=True)

        if self.get_output('w2e2_maps') != 'NONE':
            logger.info("Writing w2e2 maps to {}.".format(self.get_output('w2e2_maps')))
            header = self.fsk.wcs.to_header()
            hdus = []
            shp_mp = [self.fsk.ny, self.fsk.nx]
            for im, m_list in enumerate(w2e2maps):
                bin_tag = im + 1

                # Maps
                head = header.copy()
                head['DESCR'] = ('w2e2_1, bin {}'.format(bin_tag),
                                 'Description')
                if im == 0:
                    hdu = fits.PrimaryHDU(data=m_list[0].reshape(shp_mp),
                                          header=head)
                else:
                    hdu = fits.ImageHDU(data=m_list[0].reshape(shp_mp),
                                        header=head)

                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('w2e2_2, bin {}'.format(bin_tag), 'Description')
                hdu = fits.ImageHDU(data=m_list[1].reshape(shp_mp),
                                    header=head)
                hdus.append(hdu)

            hdulist = fits.HDUList(hdus)
            hdulist.writeto(self.get_output('w2e2_maps'), overwrite=True)

        # Plotting
        for im, m_list in enumerate(gammamaps):
            plot_map(self.config, self.fsk, m_list[0][0], 'gamma1_%d' % im)
            plot_map(self.config, self.fsk, m_list[0][1], 'gamma2_%d' % im)
            plot_map(self.config, self.fsk, m_list[1][0], 'gamma_w_%d' % im)
            plot_map(self.config, self.fsk, m_list[1][1], 'gamma_b_%d' % im)
            plot_map(self.config, self.fsk, m_list[1][2], 'gamma_c_%d' % im)
            # z = 0.5 * (pzs_cosmos[im, 0, :] + pzs_cosmos[im, 1, :])
            # nzs = [pzs_cosmos[im, 2, :]]
            # names = ['COSMOS']
            # for n in self.pdf_files.keys():
            #     nzs.append(pzs_stack[n][im, 2, :])
            #     names.append(n)
            # plot_curves(self.config, 'nz_%d' % im,
            #             z, nzs, names, xt=r'$z$', yt=r'$N(z)$')
        x = np.arange(self.nbins)
        # plot_curves(self.config, 'mhat', np.arange(self.nbins),
        #             [mhats], ['m_hat'], xt='bin', yt=r'$\hat{m}$')
        # plot_curves(self.config, 'resp', np.arange(self.nbins),
        #             [resps], ['resp'], xt='bin', yt=r'$R$')

        # Permissions on NERSC
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

if __name__ == '__main__':
    cls = PipelineStage.main()
