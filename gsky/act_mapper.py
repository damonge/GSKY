from ceci import PipelineStage
from .types import FitsFile,ASCIIFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createCountsMap
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACTMapper(PipelineStage) :
    name="ACTMapper"
    inputs=[('masked_fraction',FitsFile)]
    outputs=[('act_maps',FitsFile)]
    config_options={'act_inputs':[]}

    def check_fsks(self, fsk1, fsk2):
        if((fsk1.nx == fsk2.nx) and (fsk1.ny == fsk2.ny) and
           (fsk1.dx == fsk2.dx) and (fsk1.dy == fsk2.dy)):
            return False
        return True

    def check_sanity(self, pix):
        # Sanity checks
        # Only two edges in x
        ix_unique = np.unique(pix[:, 0])
        check_a = len(ix_unique)==2

        # Only two edges in y
        iy_unique = np.unique(pix[:, 1])
        check_b = len(iy_unique)==2

        # Right separation between edges
        nx = int(np.fabs(np.diff(ix_unique)))
        ny = int(np.fabs(np.diff(iy_unique)))
        check_c = (nx == self.fsk_hsc.nx) and (ny = self.fsk_hsc.ny)


        # Integer pixel coordinates
        check_d = np.all(np.fabs(iy_unique - np.rint(iy_unique))<1E-5)
        check_e = np.all(np.fabs(ix_unique - np.rint(ix_unique))<1E-5)

        if not (check_a * check_b * check_c * check_d * check_e):
            raise ValueError("Sanity checks don't pass")

    def compute_edges(self):

        self.coords_corner = self.fsk_hsc.wcs.all_pix2world([[0,0],
                                                             [self.fsk_hsc.nx,0],
                                                             [0,self.fsk_hsc.ny],
                                                             [self.fsk_hsc.nx,
                                                              self.fsk_hsc_ny]],0)
        pix = self.fsk_act.wcs.all_world2pix(self.coords_corner, 0)
        self.check_sanity(pix)

        self.ix0_act = int(np.amin(np.unique(pix[:,0])))
        self.ixf_act = int(np.amax(np.unique(pix[:,0])))
        self.iy0_act = int(np.amin(np.unique(pix[:,1])))
        self.iyf_act = int(np.amax(np.unique(pix[:,1])))

        # Translate in case HSC lies partially outside of ACT
        self.iyf_hsc = self.fsk_hsc.ny
        if self.iy0_act<0:
            self.iyf_hsc+=self.iy0_act
            self.iy0_act=0

        self.iy0_hsc = 0
        if self.iyf_act>self.fsk_act.ny:
            self.iy0_hsc+=self.iyf_act-self.fsk_act.ny
            self.iyf_act=self.fsk_act.ny

        self.ixf_hsc = self.fsk_hsc.nx
        if self.ix0_act<0:
            self.ixf_hsc+=self.ix0_act
            self.ix0_act=0

        self.ix0_hsc = 0
        if self.ixf_act>self.fsk_act.nx:
            self.ix0_hsc+=self.ixf_act-self.fsk_act.nx
            self.ixf_act=self.fsk_act.nx

    def process_inputs(self):
        self.fsk_hsc,_=read_flat_map(self.get_input("masked_fraction"))
        
        self.act_maps_full=[]
        self.fsk_act = None
        for d in self.config['act_inputs']:
            mdir = {}
            fskb, msk = read_flat_map(d['mask'])
            fskc, mpp = read_flat_map(d['map'])
            if self.check_fsks(fskb, fskc):
                raise ValueError("Footprints are incompatible")
            if self.fsk_act is None:
                self.fsk_act = fskb
            else:
                if self.check_fsks(fskb, self.fsk_act):
                    raise ValueError("ACT footprints are inconsistent")
            mdir['name'] = d['name']
            mdir['mask'] = mask
            mdir['map'] = mpp
            self.act_maps_full.append(mdir)

    def cut_act_map(self, mp):
        mp_out = np.zeros([self.fsk_hsc.ny, self.fsk_hsc.nx])
        mp_out[self.iy0_hsc:self.iyf_hsc,
               self.ix0_hsc:self.ixf_hsc] = mp[self.iy0_act:self.iyf_act,
                                               self.ix0_act:self.ixf_act]
        return mp_out
        
    def get_nmaps(self,cat) :
        """
        Get number counts map from catalog
        """
        maps=[]

        for i in range(self.nbins):
            msk_bin = cat['tomo_bin']==i
            subcat=cat[msk_bin]
            nmap=createCountsMap(subcat['ra'],subcat['dec'],self.fsk)
            maps.append(nmap)
        return np.array(maps)

    def get_nz_cosmos(self) :
        """
        Get N(z) from weighted COSMOS-30band data
        """
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][ 1:]

        if self.config['pz_code']=='ephor_ab' :
            pz_code='eab'
        elif self.config['pz_code']=='frankenz' :
            pz_code='frz'
        elif self.config['pz_code']=='nnpz' :
            pz_code='nnz'
        else :
            raise KeyError("Photo-z method "+self.config['pz_code']+
                           " unavailable. Choose ephor_ab, frankenz or nnpz")

        if self.config['pz_mark']  not in ['best','mean','mode','mc'] :
            raise KeyError("Photo-z mark "+self.config['pz_mark']+
                           " unavailable. Choose between best, mean, mode and mc")

        self.column_mark = 'pz_'+self.config['pz_mark']+'_'+pz_code

        weights_file=fits.open(self.get_input('cosmos_weights'))[1].data

        pzs=[]
        for zi,zf in zip(zi_arr,zf_arr) :
            msk_cosmos=(weights_file[self.column_mark]<=zf) & (weights_file[self.column_mark]>zi)
            hz,bz=np.histogram(weights_file[msk_cosmos]['PHOTOZ'],
                               bins=self.config['nz_bin_num'],
                               range=[0.,self.config['nz_bin_max']],
                               weights=weights_file[msk_cosmos]['weight'])
            hnz,bnz=np.histogram(weights_file[msk_cosmos]['PHOTOZ'],
                                 bins=self.config['nz_bin_num'],
                                 range=[0.,self.config['nz_bin_max']])
            ehz=np.zeros(len(hnz)); ehz[hnz>0]=(hz[hnz>0]+0.)/np.sqrt(hnz[hnz>0]+0.)
            pzs.append([bz[:-1],bz[1:],(hz+0.)/np.sum(hz+0.),ehz])
        return np.array(pzs)

    def get_nz_stack(self,cat,codename) :
        """
        Get N(z) from pdf stacks.
        :param cat: object catalog
        :param codename: photoz code name (demp, ephor, ephor_ab, frankenz or nnpz).
        """
        from scipy.interpolate import interp1d

        f=fits.open(self.pdf_files[codename])
        p=f[1].data['pdf'][self.msk]
        z=f[2].data['bins']
        sumpdf = np.sum(p,axis=1)
        pdfgood = sumpdf > 0

        z_all=np.linspace(0.,self.config['nz_bin_max'],self.config['nz_bin_num']+1)
        z0=z_all[:-1]; z1=z_all[1:]; zm=0.5*(z0+z1)
        pzs=[]
        for i in range(self.nbins):
            msk_good = (cat['tomo_bin']==i) & pdfgood
            hz_orig=np.sum(p[msk_good],axis=0)
            hz_orig/=np.sum(hz_orig)
            hzf=interp1d(z,hz_orig,bounds_error=False,fill_value=0.)
            hzm=hzf(zm);
            
            pzs.append([z0,z1,hzm/np.sum(hzm)])
        f.close()
        return np.array(pzs)

    def run(self) :
        """
        Main routine. This stage:
        - Creates number density maps from the reduced catalog for a set of redshift bins.
        - Calculates the associated N(z)s for each bin using different methods.
        - Stores the above into a single FITS file
        """
        logger.info("Reading masked fraction")
        self.fsk,_=read_flat_map(self.get_input("masked_fraction"))
        self.nbins = len(self.config['pz_bins'])-1

        logger.info("Reading catalog")
        cat=fits.open(self.get_input('clean_catalog'))[1].data
        #Remove masked objects
        if self.config['mask_type']=='arcturus' :
            self.msk=cat['mask_Arcturus'].astype(bool)
        elif self.config['mask_type']=='sirius' :
            self.msk=np.logical_not(cat['iflags_pixel_bright_object_center'])
            self.msk*=np.logical_not(cat['iflags_pixel_bright_object_any'])
        else :
            raise KeyError("Mask type "+self.config['mask_type']+
                           " not supported. Choose arcturus or sirius")
        self.msk *= cat['wl_fulldepth_fullcolor']
        cat=cat[self.msk]

        logger.info("Reading pdf filenames")
        data_syst=np.genfromtxt(self.get_input('pdf_matched'),
                                dtype=[('pzname','|U8'),('fname','|U256')])
        self.pdf_files={n:fn for n,fn in zip(np.atleast_1d(data_syst['pzname']),
                                             np.atleast_1d(data_syst['fname']))}
        
        logger.info("Getting COSMOS N(z)s")
        pzs_cosmos=self.get_nz_cosmos()

        logger.info("Getting pdf stacks")
        pzs_stack={}
        for n in self.pdf_files.keys() :
            pzs_stack[n]=self.get_nz_stack(cat,n)

        logger.info("Getting number count maps")
        n_maps=self.get_nmaps(cat)

        logger.info("Writing output")
        header=self.fsk.wcs.to_header()
        hdus=[]
        for im,m in enumerate(n_maps) :
            #Map
            head=header.copy()
            head['DESCR']=('Ngal, bin %d'%(im+1),'Description')
            if im==0 :
                hdu=fits.PrimaryHDU(data=m.reshape([self.fsk.ny,self.fsk.nx]),header=head)
            else :
                hdu=fits.ImageHDU(data=m.reshape([self.fsk.ny,self.fsk.nx]),header=head)
            hdus.append(hdu)
            
            #Nz
            cols=[fits.Column(name='z_i',array=pzs_cosmos[im,0,:],format='E'),
                  fits.Column(name='z_f',array=pzs_cosmos[im,1,:],format='E'),
                  fits.Column(name='nz_cosmos',array=pzs_cosmos[im,2,:],format='E'),
                  fits.Column(name='enz_cosmos',array=pzs_cosmos[im,3,:],format='E')]
            for n in self.pdf_files.keys() :
                cols.append(fits.Column(name='nz_'+n,array=pzs_stack[n][im,2,:],format='E'))
            hdus.append(fits.BinTableHDU.from_columns(cols))
        hdulist=fits.HDUList(hdus)
        hdulist.writeto(self.get_output('ngal_maps'),overwrite=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
