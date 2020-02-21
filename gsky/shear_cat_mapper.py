from ceci import PipelineStage
from .types import FitsFile,ASCIIFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createSpin2Map
from .gal_mapper import GalMapper
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShearCatMapper(GalMapper) :
    name="ShearCatMapper"
    inputs=[('calib_catalog', FitsFile), ('masked_fraction', FitsFile)]
    outputs=[('gamma_maps', FitsFile)]
    config_options={'pz_code':'ephor_ab', 'pz_mark':'best',
                    'pz_bins':[0.15,0.50,0.75,1.00,1.50], 'nz_bin_num':200,
                    'nz_bin_max':3.0, 'shearrot': 'flipu'}
    
    '''
    def _responsivity(self, cat):
        """
        Compute shear responsivity.
        For HSC (see Mandelbaum et al., 2018, arXiv:1705.06745):
        R = 1 - < e_rms^2 >w (Eq. (A1) in Mandelbaum et al., 2018)
        :param cat:
        :return:
        """

        R = 1. - np.average(cat['ishape_hsm_regauss_derived_rms_e']**2, weights=cat['ishape_hsm_regauss_derived_shape_weight'])

        return R

    def _mhat(self, cat):
        """
        Compute multiplicative bias.
        For HSC (see Mandelbaum et al., 2018, arXiv:1705.06745):
        mhat = < m >w (Eq. (A2) in Mandelbaum et al., 2018)
        :param cat:
        :return:
        """

        mhat = np.average(cat['ishape_hsm_regauss_derived_shear_bias_m'], weights=cat['ishape_hsm_regauss_derived_shape_weight'])

        return mhat

    def calibrated_catalog(self, cat, R=None, mhat=None):
        """
        Calibrate shear catalog and add calibrated shear columns to existing catalog.
        For HSC (see Mandelbaum et al., 2018, arXiv:1705.06745):
        gi = 1/(1 + mhat)[ei/(2R) - ci] (Eq. (A6) in Mandelbaum et al., 2018)
        R = 1 - < e_rms^2 >w (Eq. (A1) in Mandelbaum et al., 2018)
        mhat = < m >w (Eq. (A2) in Mandelbaum et al., 2018)
        :param cat:
        :param R:
        :param mhat:
        :return:
        """

        logger.info('Computing calibrated shear catalog.')

        cat_calib = copy.deepcopy(cat)

        if R is None and mhat is None:
            logger.info('Computing R and mhat.')
            R = self._responsivity(cat_calib)
            mhat = self._mhat(cat_calib)

        else:
            logger.info('R and mhat provided.')

        logger.info('R = {}, mhat = {}.'.format(R, mhat))

        e1_corr = 1./(1. + mhat)*(cat_calib['ishape_hsm_regauss_e1']/(2.*R) - cat_calib['ishape_hsm_regauss_derived_shear_bias_c1'])
        e2_corr = 1./(1. + mhat)*(cat_calib['ishape_hsm_regauss_e2']/(2.*R) - cat_calib['ishape_hsm_regauss_derived_shear_bias_c2'])

        # Add these two columns to catalog
        cat_calib = Table(cat_calib)
        cat_calib['ishape_hsm_regauss_e1_calib'] = e1_corr
        cat_calib['ishape_hsm_regauss_e2_calib'] = e2_corr

        logger.info('Columns ishape_hsm_regauss_e1_calib, ishape_hsm_regauss_e2_calib added to shear catalog.')

        return cat_calib, R, mhat

    def pz_cut(self, cat):
        """
        Apply pz cut to catalog.
        :param cat:
        :return:
        """

        logger.info('Applying pz cut to catalog. Using {} with zmin = {}, zmax = {}.'.\
                    format(self.config['photoz_method'], self.config['photoz_min'], self.config['photoz_max']))

        photozmask = (cat[self.config['photoz_method']]>=self.config['photoz_min'])\
                     &(cat[self.config['photoz_method']]<self.config['photoz_max'])

        cat = copy.deepcopy(cat)
        cat = cat[photozmask]

        return cat
    '''
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

        self.parse_input()

        logger.info("Reading masked fraction from {}.".format(self.get_input("masked_fraction")))
        self.fsk, _ = read_flat_map(self.get_input("masked_fraction"))

        logger.info("Reading calibrated shear catalog from {}.".format(self.get_input('calib_catalog')))
        cat = fits.open(self.get_input('calib_catalog'))[1].data

        #logger.info("Reading pdf filenames.")
        #data_syst = np.genfromtxt(self.get_input('pdf_matched'),
        #                          dtype=[('pzname', '|U8'), ('fname', '|U256')])
        #self.pdf_files = {n: fn for n, fn in zip(data_syst['pzname'], data_syst['fname'])}

        logger.info("Parsing photo-z bins.")
        self.zi_arr = self.config['pz_bins'][:-1]
        self.zf_arr = self.config['pz_bins'][1:]
        self.nbins = len(self.zi_arr)

        #logger.info("Getting COSMOS N(z)s.")
        #pzs_cosmos = self.get_nz_cosmos()

        #logger.info("Getting pdf stacks.")
        #pzs_stack = {}
        #for n in self.pdf_files.keys():
        #    pzs_stack[n] = self.get_nz_stack(cat, n)

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
                hdu = fits.PrimaryHDU(data=m_list[0][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma2, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[0][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma weight mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[1][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('gamma binary mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[1][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('counts map (shear sample), bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[1][2].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
            else:
                head = header.copy()
                head['DESCR'] = ('gamma1, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[0][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma2, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[0][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = ('gamma weight mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[1][0].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('gamma binary mask, bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[1][1].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)
                head['DESCR'] = ('counts map (shear sample), bin %d'%(im+1), 'Description')
                hdu = fits.ImageHDU(data=m_list[1][2].reshape([self.fsk.ny,self.fsk.nx]), header=head)
                hdus.append(hdu)

            # e2rms
            cols = [fits.Column(name='e2rms', array=e2rms[im], format='E')]
            hdus.append(fits.BinTableHDU.from_columns(cols))

            # Nz
            cols = []
            #cols=[fits.Column(name='z_i',array=pzs_cosmos[im,0,:],format='E'),
            #      fits.Column(name='z_f',array=pzs_cosmos[im,1,:],format='E'),
            #      fits.Column(name='nz_cosmos',array=pzs_cosmos[im,2,:],format='E'),
            #      fits.Column(name='enz_cosmos',array=pzs_cosmos[im,3,:],format='E')]
            #for n in self.pdf_files.keys() :
            #    cols.append(fits.Column(name='nz_'+n,array=pzs_stack[n][im,2,:],format='E'))
            #hdus.append(fits.BinTableHDU.from_columns(cols))

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.get_output('gamma_maps'), overwrite=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
