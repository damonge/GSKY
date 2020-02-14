from ceci import PipelineStage
from .types import FitsFile,ASCIIFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createSpin2Map
from .cat_mapper import CatMapper
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShearCatMapper(CatMapper) :
    name="ShearCatMapper"
    inputs=[('clean_catalog', FitsFile), ('masked_fraction', FitsFile)]
    outputs=[('gamma_maps', FitsFile)]
    config_options={'pz_code':'ephor_ab', 'pz_mark':'best',
                    'pz_bins':[0.15,0.50,0.75,1.00,1.50], 'nz_bin_num':200,
                    'nz_bin_max':3.0, 'shearrot': 'flipuq'}
    
    def get_gamma_maps(self, cat):
        """
        Get gamma1, gamma2 maps and corresponding mask from catalog.
        :param cat:
        :return:
        """

        if not 'ishape_hsm_regauss_e1_calib' in cat.dtype.names:
            raise RuntimeError('get_gamma_maps must be called with calibrated shear catalog. Aborting.')
        maps = []

        for zi, zf in zip(self.zi_arr, self.zf_arr) :
            msk_bin = (cat[self.column_mark]<=zf) & (cat[self.column_mark]>zi)
            subcat = cat[msk_bin]
            gammamaps, gammamasks = createSpin2Map(subcat['ra'], subcat['dec'], subcat['ishape_hsm_regauss_e1_calib'], \
                                     subcat['ishape_hsm_regauss_e2_calib'], self.fsk, \
                                     weights=subcat['ishape_hsm_regauss_derived_shape_weight'], \
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

        if not 'ishape_hsm_regauss_e1_calib' in cat.dtype.names:
            raise RuntimeError('get_e2rms must be called with calibrated shear catalog. Aborting.')
        e2rms_arr = []

        for zi, zf in zip(self.zi_arr, self.zf_arr) :
            msk_bin = (cat[self.column_mark]<=zf) & (cat[self.column_mark]>zi)
            subcat = cat[msk_bin]
            e1_2rms = np.average(subcat['ishape_hsm_regauss_e1_calib']**2,
                                 weights=subcat['ishape_hsm_regauss_derived_shape_weight'])
            e2_2rms = np.average(subcat['ishape_hsm_regauss_e2_calib'] ** 2,
                                 weights=subcat['ishape_hsm_regauss_derived_shape_weight'])

            e2rms_combined = np.array([e1_2rms, e2_2rms])
            e2rms_arr.append(e2rms_combined)

        return e2rms_arr

    def get_nz_stack(self, cat, codename):
        """
        Get N(z) from pdf stacks.
        :param cat: object catalog
        :param codename: photoz code name (demp, ephor, ephor_ab, frankenz or nnpz).
        """
        logger.info("Creating pdf stacks for cosmic shear.")

        from scipy.interpolate import interp1d

        f = fits.open(self.pdf_files[codename])
        p = f[1].data['pdf']
        z = f[2].data['bins']

        z_all = np.linspace(0., self.config['nz_bin_max'], self.config['nz_bin_num'] + 1)
        z0 = z_all[:-1]
        z1 = z_all[1:]
        zm = 0.5 * (z0 + z1)
        pzs = []
        for zi, zf in zip(self.zi_arr, self.zf_arr):
            msk_bin = (cat[self.column_mark] <= zf) & (cat[self.column_mark] > zi)
            logger.info("Weighing pdf by WL shape weight.")
            hz_orig = np.sum(cat['ishape_hsm_regauss_derived_shape_weight'][:, np.newaxis]*p[msk_bin], axis=0)
            hz_orig /= np.sum(hz_orig)
            hzf = interp1d(z, hz_orig, bounds_error=False, fill_value=0.)
            hzm = hzf(zm)

            pzs.append([z0, z1, hzm / np.sum(hzm)])

        return np.array(pzs)

    def run(self):
        """
        Main routine. This stage:
        - Creates gamma1, gamma2 maps and corresponding masks from the reduced catalog for a set of redshift bins.
        - Stores the above into a single FITS file.
        """

        logger.info("Reading masked fraction from {}.".format(self.get_input("masked_fraction")))
        self.fsk, _ = read_flat_map(self.get_input("masked_fraction"))

        logger.info("Reading calibrated shear catalog from {}.".format(self.get_input('clean_catalog')))
        cat = fits.open(self.get_input('clean_catalog'))[1].data

        logger.info("Reading pdf filenames.")
        data_syst = np.genfromtxt(self.get_input('pdf_matched'),
                                  dtype=[('pzname', '|U8'), ('fname', '|U256')])
        self.pdf_files = {n: fn for n, fn in zip(data_syst['pzname'], data_syst['fname'])}

        logger.info("Parsing photo-z bins.")
        self.zi_arr = self.config['pz_bins'][:-1]
        self.zf_arr = self.config['pz_bins'][1:]
        self.nbins = len(self.zi_arr)

        logger.info("Getting COSMOS N(z)s.")
        pzs_cosmos = self.get_nz_cosmos()

        logger.info("Getting pdf stacks.")
        pzs_stack = {}
        for n in self.pdf_files.keys():
            pzs_stack[n] = self.get_nz_stack(cat, n)

        logger.info("Creating shear maps and corresponding masks.")
        gammamaps = self.get_gamma_maps(cat)

        logger.info("Computing e2rms.")
        e2rms = self.get_e2rms(cat)

        print("Writing output to {}.".format(self.get_output('gamma_maps')))
        header = self.fsk.wcs.to_header()
        hdus = []
        for im, m_list in enumerate(gammamaps) :
            # Maps
            if im == 0 :
                head = header.copy()
                head['DESCR'] = ('gamma1, bin %d'%(im+1), 'Description')
                hdu = fits.PrimaryHDU(data=m_list[im][0][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma2, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][0][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma weight mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][1][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('gamma binary mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][1][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('counts map (shear sample), bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][1][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
            else:
                head = header.copy()
                head['DESCR'] = ('gamma1, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][0][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma2, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][0][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma weight mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][1][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('gamma binary mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][1][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('counts map (shear sample), bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[im][1][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)

            # e2rms
            cols = [fits.Column(name='e2rms', array=e2rms[im], format='E')]
            hdus.append(fits.BinTableHDU.from_columns(cols))

            # Nz
            cols=[fits.Column(name='z_i',array=pzs_cosmos[im,0,:],format='E'),
                  fits.Column(name='z_f',array=pzs_cosmos[im,1,:],format='E'),
                  fits.Column(name='nz_cosmos',array=pzs_cosmos[im,2,:],format='E'),
                  fits.Column(name='enz_cosmos',array=pzs_cosmos[im,3,:],format='E')]
            for n in self.pdf_files.keys() :
                cols.append(fits.Column(name='nz_'+n,array=pzs_stack[n][im,2,:],format='E'))
            hdus.append(fits.BinTableHDU.from_columns(cols))

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.get_output('gamma_maps'), overwrite=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
