import numpy as np

import copy
from astropy.table import Table, vstack
from astropy.io import fits
from ceci import PipelineStage
from .types import FitsFile, ASCIIFile

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReduceShearCat(PipelineStage):

    name = "ReduceShearCat"
    inputs = [('clean_catalog', FitsFile)]
    outputs = [('shear_catalog', FitsFile)]
    config_options = {}

    def shear_cut(self, cat):
        """
        Apply additional shear cuts to catalog.
        :param cat:
        :return:
        """

        logger.info('Applying shear cuts to catalog.')

        ishape_flags_mask = cat['ishape_hsm_regauss_flags'] == False
        ishape_sigma_mask = ~np.isnan(cat['ishape_hsm_regauss_sigma'])
        ishape_resolution_mask = cat['ishape_hsm_regauss_resolution'] >= 0.3
        ishape_shear_mod_mask = (cat['ishape_hsm_regauss_e1']**2 + cat['ishape_hsm_regauss_e2']**2) < 2
        ishape_sigma_mask *= (cat['ishape_hsm_regauss_sigma'] >= 0.)*(cat['ishape_hsm_regauss_sigma'] <= 0.4)

        shearmask = ishape_flags_mask*ishape_sigma_mask*ishape_resolution_mask*ishape_shear_mod_mask
        return shearmask

    def run(self) :
        """
        Main function.
        This stage:
        - Reduces shear catalog.
        """

        #Read catalog
        logger.info('Reading cleaned catalog from {}.'.format(self.get_input('clean_catalog')))
        cat = fits.open(self.get_input('clean_catalog'))[1].data

        logger.info('Initial catalog size: {}.'.format(len(cat)))
        sel = self.shear_cut(cat)
        cat = Table(cat)
        cat.remove_rows(~sel)
        logger.info('Catalog size after shear cut: {}.'.format(len(cat)))

        ####
        # Write shear catalog
        # 1- header
        logger.info('Writing shear catalog.')
        hdr = fits.Header()
        hdr['CATALOG']='shear'
        prm_hdu = fits.PrimaryHDU(header=hdr)
        # 2- Catalog
        cat_hdu = fits.table_to_hdu(cat)
        # 3- Actual writing
        hdul = fits.HDUList([prm_hdu,cat_hdu])
        hdul.writeto(self.get_output('shear_catalog'), overwrite=True)
        ####

if __name__ == '__main__':
    cls = PipelineStage.main()
