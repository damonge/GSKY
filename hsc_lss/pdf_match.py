from ceci import PipelineStage
from .types import FitsFile,ASCIIFile
from astropy.table import Table
import numpy as np
from astropy.io import fits
import os
import pandas as pd

class PDFMatch(PipelineStage) :
    name="PDFMatch"
    inputs=[('clean_catalog',FitsFile),('pdf_dir',None)]
    outputs=[('pdf_matched',ASCIIFile)]
    config_options={}

    def run(self) :
        """
        Main function.
        This stage matches each object in the reduced catalog with its photo-z pdf for different
        photo-z codes. Then stores the matched pdfs with the same ordering as the reduced catalog
        into a separate FITS file.
        """
        file_out=self.get_output('pdf_matched')
        prefix_out=file_out[:-4]
        pz_algs=['demp','ephor','ephor_ab','frankenz','nnpz']

        str_out=""

        #Read catalog
        cat=Table.read(self.get_input('clean_catalog'),format='fits')
        hscdata=cat.to_pandas()

        #Read pdfs from frames
        for alg in pz_algs :
            filename=prefix_out+"_"+alg+".fits"
            str_out+=alg+" "+filename+"\n"
            if os.path.isfile(filename) :
                print(alg+" found")
                continue

            pdfs_path=self.get_input('pdf_dir')+'/'+alg
            patch_files=[f for f in os.listdir(pdfs_path) if f.__contains__('.fits')]
            for i,file in enumerate(patch_files) :
                print("Reading %s/%s"%(pdfs_path,file))
                hdul=fits.open('%s/%s'%(pdfs_path,file))
                data_readin=np.array(hdul[1].data) # pdfs
                bins_readin=np.array(hdul[2].data) # z_bins
                # find the column numbers for ids, pdfs
                id_ind = np.where(np.array(data_readin.dtype.names)=='ID')[0][0]
                pdf_ind = np.where(np.array(data_readin.dtype.names)=='PDF')[0][0]
                # now need to restructure the data
                if (i==0):  # first patch
                    bins = []
                    pdfs = {}

                bins_ = []
                for j in range(len(bins_readin)):
                    bins_.append(bins_readin[j][0])
                if i==0:
                    bins = bins_
                else:
                    if bins != bins_:
                        raise ValueError('Bins dont match: %s vs. %s %d'%(len(bins), len(bins_),i))

                for i in range(len(data_readin)):
                    pdfs[data_readin[i][id_ind]] = data_readin[i][pdf_ind]

            bins = np.array(bins)

            # match using IDs
            print('\nMatching IDs now ... ')
            matched_ids, matched_pdfs_cat = [], []

            for i, objID in enumerate(hscdata['object_id']):
                if objID not in pdfs.keys():
                    print('\n%s not in the chosen pdfs'%objID)
                else:
                    matched_ids.append(objID)
                    matched_pdfs_cat.append(pdfs[objID])

            # save data
            # restructure the data
            df = pd.DataFrame(matched_pdfs_cat)
            matched_pdfs = df.values

            if np.shape(matched_pdfs)[1]!=len(bins):
                raise ValuerEror('Somethings wrong. Have %s columns in matched_pdfs and %s bins.'%(np.shape(matched_pdfs)[1],
                                                                                                   len(bins)))
            # set up the header
            hdr = fits.Header()
            primary_hdu = fits.PrimaryHDU(header=hdr)

            # data to save
            # one table for pdfs and object ids
            col1 = fits.Column(name='object_id', format='K', array=np.array(matched_ids, dtype=int))
            col2 = fits.Column(name='pdf', format='%iE'%len(bins), array=matched_pdfs)
            cols = fits.ColDefs([col1, col2])
            pdf_hdu = fits.BinTableHDU.from_columns(cols)

            # a separate table for bins
            bincol = fits.Column(name='bins', format='E', array=np.array(bins, dtype=float))
            bincols = fits.ColDefs([bincol])
            bin_hdu = fits.BinTableHDU.from_columns(bincols)

            # save it
            hdul = fits.HDUList([primary_hdu, pdf_hdu, bin_hdu])
            filename=prefix_out+"_"+alg+".fits"
            hdul.writeto(filename, overwrite=True)
            
            print('\nSaved %s'%filename)

        print("Printing summary file")
        f=open(file_out,"w")
        f.write(str_out)
        f.close()

if __name__ == '__main__':
    cls = PipelineStage.main()
