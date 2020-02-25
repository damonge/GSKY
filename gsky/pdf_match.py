from ceci import PipelineStage
from .types import FitsFile, ASCIIFile
from astropy.table import Table
import numpy as np
from astropy.io import fits
import os
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFMatch(PipelineStage):
    name = "PDFMatch"
    inputs = [('clean_catalog', FitsFile),
              ('pdf_dir', None)]
    outputs = [('pdf_matched', ASCIIFile)]
    config_options = {}

    def run(self):
        """
        Main function.
        This stage matches each object in the reduced catalog
        with its photo-z pdf for different photo-z codes. Then
        stores the matched pdfs with the same ordering as the
        reduced catalog into a separate FITS file.
        """
        file_out = self.get_output('pdf_matched')
        prefix_out = self.get_output('pdf_matched', final_name=True)[:-4]
        pz_algs = ['demp', 'ephor', 'ephor_ab', 'frankenz', 'nnpz']

        str_out = ""

        # Read catalog
        cat = Table.read(self.get_input('clean_catalog'), format='fits')
        gal_ids = cat['object_id']

        # Read pdfs from frames
        for alg in pz_algs:
            filename = prefix_out+"_"+alg+".fits"
            str_out += alg+" "+filename+"\n"
            if os.path.isfile(filename):
                logger.info(alg+" found")
                continue

            pdfs_path = self.get_input('pdf_dir')+'/'+alg

            fname_bins = pdfs_path + "/pz_pdf_bins.fits"
            hdul = fits.open(fname_bins)
            bins = hdul[1].data
            hdul.close()

            matched = np.zeros(len(cat), dtype=bool)
            pdfs = np.zeros([len(cat), len(bins)])-1

            patch_files = [f for f in os.listdir(pdfs_path) if f.__contains__(alg+'.fits')]
            for i, file in enumerate(patch_files):
                logger.info("Reading %s/%s" % (pdfs_path, file))
                hdul = fits.open('%s/%s' % (pdfs_path, file))
                data_pdf = np.array(hdul[1].data)  # pdfs
                pdf_ids = data_pdf['object_id']
                ids, i_g, i_p = np.intersect1d(gal_ids, pdf_ids,
                                               assume_unique=True,
                                               return_indices=True)
                matched[i_g] = True
                pdfs[i_g, :] = data_pdf['P(z)'][i_p]

            logger.info("%d galaxies matched with pdfs out of %d" %
                        (np.sum(matched), len(matched)))
            df = pd.DataFrame(pdfs)
            matched_pdfs = df.values
            if np.shape(matched_pdfs)[1] != len(bins):
                raise ValueError('Somethings wrong. ')

            # set up the header
            hdr = fits.Header()
            primary_hdu = fits.PrimaryHDU(header=hdr)

            # data to save
            # one table for pdfs and object ids
            col1 = fits.Column(name='object_id', format='K',
                               array=np.array(gal_ids, dtype=int))
            col2 = fits.Column(name='pdf', format='%iE' % len(bins),
                               array=matched_pdfs)
            cols = fits.ColDefs([col1, col2])
            pdf_hdu = fits.BinTableHDU.from_columns(cols)

            # a separate table for bins
            bincol = fits.Column(name='bins', format='E',
                                 array=np.array(bins, dtype=float))
            bincols = fits.ColDefs([bincol])
            bin_hdu = fits.BinTableHDU.from_columns(bincols)

            # save it
            hdul = fits.HDUList([primary_hdu, pdf_hdu, bin_hdu])
            filename = prefix_out+"_"+alg+".fits"
            logger.info("Writing to file "+filename)
            hdul.writeto(filename, overwrite=True)
            logger.info('Saved %s' % filename)

        logger.info("Printing summary file")
        f = open(file_out, "w")
        f.write(str_out)
        f.close()


if __name__ == '__main__':
    cls = PipelineStage.main()
