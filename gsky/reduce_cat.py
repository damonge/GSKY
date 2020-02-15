from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table,vstack
import numpy as np
from .flatmaps import FlatMapInfo
from .map_utils import createCountsMap, createMeanStdMaps, createMask, removeDisconnected
from .estDepth import get_depth
from astropy.io import fits

class ReduceCat(PipelineStage) :
    name="ReduceCat"
    inputs=[('raw_data',None)]
    outputs=[('clean_catalog',FitsFile),('dust_map',FitsFile),('star_map',FitsFile),
             ('bo_mask',FitsFile),('masked_fraction',FitsFile),('depth_map',FitsFile)]
    config_options={'min_snr':10.,'depth_cut':24.5,'res':0.0285,
                    'res_bo':0.003,'pad':0.1,'band':'i','depth_method':'fluxerr',
                    'flat_project':'CAR','mask_type':'sirius', 'ra': 'ra', 'dec': 'dec'}
    bands=['g','r','i','z','y']

    def make_dust_map(self,cat,fsk) :
        """
        Produces a dust absorption map for each band.
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry of the output map
        """
        print("Creating dust map")
        dustmaps=[]
        dustdesc=[]
        for b in self.bands :
            m,s=createMeanStdMaps(cat[self.config['ra']],cat[self.config['dec']],cat['a_'+b],fsk)
            dustmaps.append(m)
            dustdesc.append('Dust, '+b+'-band')
        return dustmaps,dustdesc

    def make_star_map(self,cat,fsk,sel) :
        """
        Produces a star density map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry of the output map
        :param sel: mask used to select the stars to be used.
        """
        print("Creating star map")
        mstar=createCountsMap(cat[self.config['ra']][sel],cat[self.config['dec']][sel],fsk)+0.
        descstar='Stars, '+self.config['band']+'<%.2lf'%(self.config['depth_cut'])
        return mstar,descstar

    def make_bo_mask(self,cat,fsk) :
        """
        Produces a bright object mask
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry of the output map
        """
        print("Generating bright-object mask")
        if self.config['mask_type']=='arcturus' :
            flags_mask=[~cat['mask_Arcturus'].astype(bool)]
        elif self.config['mask_type']=='sirius' :
            flags_mask=[cat['iflags_pixel_bright_object_center'],
                        cat['iflags_pixel_bright_object_any']]
        else :
            raise ValueError('Mask type '+self.config['mask_type']+' not supported')
        mask_bo,fsg=createMask(cat[self.config['ra']],cat[self.config['dec']],flags_mask,fsk,self.config['res_bo'])
        return mask_bo,fsg

    def make_masked_fraction(self,cat,fsk) :
        """
        Produces a masked fraction map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry of the output map
        """
        print("Generating masked fraction map")
        masked=np.ones(len(cat))
        if self.config['mask_type']=='arcturus' :
            masked*=cat['mask_Arcturus']
        elif self.config['mask_type']=='sirius' :
            masked*=np.logical_not(cat['iflags_pixel_bright_object_center'])
            masked*=np.logical_not(cat['iflags_pixel_bright_object_any'])
        else :
            raise ValueError('Mask type '+self.config['mask_type']+' not supported')
        masked_fraction,_=createMeanStdMaps(cat[self.config['ra']],cat[self.config['dec']],masked,fsk)
        masked_fraction_cont=removeDisconnected(masked_fraction,fsk)
        return masked_fraction_cont

    def make_depth_map(self,cat,fsk) :
        """
        Produces a depth map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the geometry of the output map
        """
        print("Creating depth maps")
        method=self.config['depth_method']
        band=self.config['band']
        snrs=cat['%scmodel_flux'%band]/cat['%scmodel_flux_err'%band]
        if method=='fluxerr' :
            arr1=cat['%scmodel_flux_err'%band]
            arr2=None
        else :
            arr1=cat['%scmodel_mag'%band]
            arr2=snrs
        depth,_=get_depth(method,cat[self.config['ra']],cat[self.config['dec']],band,
                          arr1=arr1,arr2=arr2,
                          flatSkyGrid=fsk,SNRthreshold=self.config['min_snr'])
        desc='%d-s depth, '%(self.config['min_snr'])+band+' '+method+' mean'
        return depth,desc

    def run(self) :
        """
        Main function.
        This stage:
        - Reduces the raw catalog by imposing quality cuts, a cut on i-band magnitude and a star-galaxy separation cat.
        - Produces mask maps, dust maps, depth maps and star density maps.
        """
        band=self.config['band']

        #Read list of files
        f=open(self.get_input('raw_data'))
        files=[s.strip() for s in f.readlines()]
        f.close()

        #Read catalog
        cat=Table.read(files[0])
        if len(cat)>1 :
            for fname in files[1:] :
                c=Table.read(fname)
                cat=vstack([cat,c],join_type='exact')

        if band not in self.bands :
            raise ValueError("Band "+band+" not available")

        print('Initial catalog size: %d'%(len(cat)))
            
        # Clean nulls and nans
        print("Basic cleanup")
        sel=np.ones(len(cat),dtype=bool)
        names=[n for n in cat.keys()]
        isnull_names=[]
        for key in cat.keys() :
            if key.__contains__('isnull') :
                sel[cat[key]]=0
                isnull_names.append(key)
            else :
                if not key.startswith("pz_") : #Keep photo-z's even if they're NaNs
                    sel[np.isnan(cat[key])]=0
        print("Will drop %d rows"%(len(sel)-np.sum(sel)))
        cat.remove_columns(isnull_names)
        cat.remove_rows(~sel)

        fsk=FlatMapInfo.from_coords(cat[self.config['ra']],cat[self.config['dec']],self.config['res'],
                                    pad=self.config['pad']/self.config['res'],
                                    projection=self.config['flat_project'])

        #Collect sample cuts
        sel_maglim=np.ones(len(cat),dtype=bool);
        sel_maglim[cat['%scmodel_mag'%band]-
                   cat['a_%s'%band]>self.config['depth_cut']]=0
        # Blending
        sel_blended=np.ones(len(cat),dtype=bool);
        sel_blended[cat['iblendedness_abs_flux']>=0.42169650342]=0 #abs_flux<10^-0.375
        # S/N in i
        sel_fluxcut_i=np.ones(len(cat),dtype=bool);
        sel_fluxcut_i[cat['icmodel_flux']<10*cat['icmodel_flux_err']]=0
        # S/N in g
        sel_fluxcut_g=np.ones(len(cat),dtype=int);
        sel_fluxcut_g[cat['gcmodel_flux']<5*cat['gcmodel_flux_err']]=0
        # S/N in r
        sel_fluxcut_r=np.ones(len(cat),dtype=int);
        sel_fluxcut_r[cat['rcmodel_flux']<5*cat['rcmodel_flux_err']]=0
        # S/N in z
        sel_fluxcut_z=np.ones(len(cat),dtype=int);
        sel_fluxcut_z[cat['zcmodel_flux']<5*cat['zcmodel_flux_err']]=0
        # S/N in y
        sel_fluxcut_y=np.ones(len(cat),dtype=int);
        sel_fluxcut_y[cat['ycmodel_flux']<5*cat['ycmodel_flux_err']]=0
        # S/N in grzy (at least 2 pass)
        sel_fluxcut_grzy=(sel_fluxcut_g+sel_fluxcut_r+sel_fluxcut_z+sel_fluxcut_y>=2)
        # Overall S/N
        sel_fluxcut=sel_fluxcut_i*sel_fluxcut_grzy
        # Stars
        sel_stars=np.ones(len(cat),dtype=bool);
        sel_stars[cat['iclassification_extendedness']>0.99]=0
        # Galaxies
        sel_gals =np.ones(len(cat),dtype=bool);
        sel_gals[cat['iclassification_extendedness']<0.99]=0

        ####
        # Generate systematics maps
        # 1- Dust
        dustmaps,dustdesc=self.make_dust_map(cat,fsk)
        fsk.write_flat_map(self.get_output('dust_map'),np.array(dustmaps),descript=dustdesc)

        # 2- Nstar
        #    This needs to be done for stars passing the same cuts as the sample 
        #    (except for the s/g separator)
        # Above magnitude limit
        mstar,descstar=self.make_star_map(cat,fsk,sel_maglim*sel_stars*sel_fluxcut*sel_blended)
        fsk.write_flat_map(self.get_output('star_map'),mstar,descript=descstar)

        
        #Binary BO mask
        mask_bo,fsg=self.make_bo_mask(cat,fsk)
        fsg.write_flat_map(self.get_output('bo_mask'),mask_bo,descript='Bright-object mask')

        #Masked fraction
        masked_fraction_cont=self.make_masked_fraction(cat,fsk)
        fsk.write_flat_map(self.get_output('masked_fraction'),masked_fraction_cont,
                           descript='Masked fraction')

        ####
        # Compute depth map
        depth,desc=self.make_depth_map(cat,fsk)
        fsk.write_flat_map(self.get_output('depth_map'),depth,descript=desc)

        ####
        # Implement final cuts
        # - Mag. limit
        # - S/N cut
        # - Star-galaxy separator
        # - Blending
        sel=~(sel_maglim*sel_gals*sel_fluxcut*sel_blended)
        print("Will lose %d objects to depth, S/N and stars"%(np.sum(sel)))
        cat.remove_rows(sel)

        ####
        # Write final catalog
        # 1- header
        print("Writing output")
        hdr=fits.Header()
        hdr['BAND']=self.config['band']
        hdr['DEPTH']=self.config['depth_cut']
        prm_hdu=fits.PrimaryHDU(header=hdr)
        # 2- Catalog
        cat_hdu=fits.table_to_hdu(cat)
        # 3- Actual writing
        hdul=fits.HDUList([prm_hdu,cat_hdu])
        hdul.writeto(self.get_output('clean_catalog'),overwrite=True)

        ####

if __name__ == '__main__':
    cls = PipelineStage.main()
