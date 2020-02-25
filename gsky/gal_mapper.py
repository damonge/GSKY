from ceci import PipelineStage
from .types import FitsFile, ASCIIFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createCountsMap
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GalMapper(PipelineStage):
    name = "GalMapper"
    inputs = [('clean_catalog', FitsFile), ('masked_fraction', FitsFile),
              ('cosmos_weights', FitsFile), ('pdf_matched', ASCIIFile)]
    outputs = [('ngal_maps', FitsFile)]
    config_options = {'mask_type': 'sirius',
                      'pz_code': 'ephor_ab',
                      'pz_mark': 'best',
                      'pz_bins': [0.15, 0.50, 0.75, 1.00, 1.50],
                      'nz_bin_num': 200,
                      'nz_bin_max': 3.0}

    def get_nmaps(self, cat):
        """
        Get number counts map from catalog
        """
        maps = []

        for i in range(self.nbins):
            msk_bin = cat['tomo_bin'] == i
            subcat = cat[msk_bin]
            nmap = createCountsMap(subcat['ra'], subcat['dec'], self.fsk)
            maps.append(nmap)
        return np.array(maps)

    def get_nz_cosmos(self):
        """
        Get N(z) from weighted COSMOS-30band data
        """
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][1:]

        if self.config['pz_code'] == 'ephor_ab':
            pz_code = 'eab'
        elif self.config['pz_code'] == 'frankenz':
            pz_code = 'frz'
        elif self.config['pz_code'] == 'nnpz':
            pz_code = 'nnz'
        else:
            raise KeyError("Photo-z method "+self.config['pz_code'] +
                           " unavailable. Choose ephor_ab, frankenz or nnpz")

        if self.config['pz_mark'] not in ['best', 'mean', 'mode', 'mc']:
            raise KeyError("Photo-z mark "+self.config['pz_mark'] +
                           " unavailable. Choose between best, mean, " +
                           "mode and mc")

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

        z_all = np.linspace(0., self.config['nz_bin_max'],
                            self.config['nz_bin_num']+1)
        z0 = z_all[:-1]
        z1 = z_all[1:]
        zm = 0.5*(z0+z1)
        pzs = []
        for i in range(self.nbins):
            msk_good = (cat['tomo_bin'] == i) & pdfgood
            hz_orig = np.sum(p[msk_good], axis=0)
            hz_orig /= np.sum(hz_orig)
            hzf = interp1d(z, hz_orig, bounds_error=False,
                           fill_value=0.)
            hzm = hzf(zm)

            pzs.append([z0, z1, hzm/np.sum(hzm)])
        f.close()
        return np.array(pzs)

    def run(self):
        """
        Main routine. This stage:
        - Creates number density maps from the reduced catalog for
          a set of redshift bins.
        - Calculates the associated N(z)s for each bin using
          different methods.
        - Stores the above into a single FITS file
        """
        logger.info("Reading masked fraction")
        self.fsk, _ = read_flat_map(self.get_input("masked_fraction"))
        self.nbins = len(self.config['pz_bins'])-1

        logger.info("Reading catalog")
        cat = fits.open(self.get_input('clean_catalog'))[1].data
        # Remove masked objects
        if self.config['mask_type'] == 'arcturus':
            self.msk = cat['mask_Arcturus'].astype(bool)
        elif self.config['mask_type'] == 'sirius':
            self.msk = np.logical_not(cat['iflags_pixel_bright_object_center'])
            self.msk *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        else:
            raise KeyError("Mask type "+self.config['mask_type'] +
                           " not supported. Choose arcturus or sirius")
        self.msk *= cat['wl_fulldepth_fullcolor']
        cat = cat[self.msk]

        logger.info("Reading pdf filenames")
        data_syst = np.genfromtxt(self.get_input('pdf_matched'),
                                  dtype=[('pzname', '|U8'),
                                         ('fname', '|U256')])
        self.pdf_files = {n: fn
                          for n, fn in zip(np.atleast_1d(data_syst['pzname']),
                                           np.atleast_1d(data_syst['fname']))}

        logger.info("Getting COSMOS N(z)s")
        pzs_cosmos = self.get_nz_cosmos()

        logger.info("Getting pdf stacks")
        pzs_stack = {}
        for n in self.pdf_files.keys():
            pzs_stack[n] = self.get_nz_stack(cat, n)

        logger.info("Getting number count maps")
        n_maps = self.get_nmaps(cat)

        logger.info("Writing output")
        header = self.fsk.wcs.to_header()
        hdus = []
        for im, m in enumerate(n_maps):
            # Map
            head = header.copy()
            head['DESCR'] = ('Ngal, bin %d' % (im+1),
                             'Description')
            if im == 0:
                hdu = fits.PrimaryHDU(data=m.reshape([self.fsk.ny,
                                                      self.fsk.nx]),
                                      header=head)
            else:
                hdu = fits.ImageHDU(data=m.reshape([self.fsk.ny,
                                                    self.fsk.nx]),
                                    header=head)
            hdus.append(hdu)

            # Nz
            cols = [fits.Column(name='z_i', array=pzs_cosmos[im, 0, :],
                                format='E'),
                    fits.Column(name='z_f', array=pzs_cosmos[im, 1, :],
                                format='E'),
                    fits.Column(name='nz_cosmos', array=pzs_cosmos[im, 2, :],
                                format='E'),
                    fits.Column(name='enz_cosmos', array=pzs_cosmos[im, 3, :],
                                format='E')]
            for n in self.pdf_files.keys():
                cols.append(fits.Column(name='nz_'+n,
                                        array=pzs_stack[n][im, 2, :],
                                        format='E'))
            hdus.append(fits.BinTableHDU.from_columns(cols))
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.get_output('ngal_maps'), overwrite=True)

        # Plotting
        for im, m in enumerate(n_maps):
            plot_map(self.config, fsk, m, 'ngal_%d' % im)
            z = 0.5 * (pzs_cosmos[im, 0, :] + pzs_cosmos[im, 1, :])
            nzs = [pzs_cosmos[im, 2, :]]
            names = ['COSMOS']
            for n in self.pdf_files.keys():
                nzs.append(pzs_stacks[n][im, 2, :])
                names.append(n)
            plot_curves(self.config, 'nz_%d' % im,
                        z, nzs, names, xt=r'$z$', yt=r'$N(z)$')


if __name__ == '__main__':
    cls = PipelineStage.main()
