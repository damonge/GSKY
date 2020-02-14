from ceci import PipelineStage
from .types import FitsFile,ASCIIFile,DirFile
import numpy as np
from .flatmaps import read_flat_map, compare_infos
from scipy.stats import binned_statistic
from astropy.io import fits
import matplotlib.pyplot as plt
import os

class MapDiagnoser(PipelineStage) :
    name="MapDiagnoser"
    inputs=[('masked_fraction',FitsFile),('ngal_maps',FitsFile),
            ('dust_map',FitsFile),('star_map',FitsFile),('depth_map',FitsFile),
            ('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),('exptime_maps',FitsFile),
            ('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),('seeing_maps',FitsFile),
            ('ellipt_maps',FitsFile),('nvisit_maps',FitsFile)]
    outputs=[('systmap_plots',DirFile)]
    config_options={'nbins_syst':10, 'n_jk':50}

    def compute_stats(self,ng_map,sys_map):
        mask=self.mskfrac*self.msk_bi
        binmask=mask>0

        # N_g/<N_g>
        ng_mean=np.sum(ng_map[binmask])/np.sum(mask[binmask])
        ng_use=ng_map[binmask]/(ng_mean*mask[binmask])
        # S/<S>
        sys_mean=np.mean(sys_map[binmask])
        sys_use=sys_map[binmask]/sys_mean

        # Compute mean number density as a function of systematic
        mean, bin_edges, _  = binned_statistic(sys_use, ng_use, statistic='mean',
                                               bins=self.config['nbins_syst'])
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        # Compute uncertainties through jackknife
        means = np.zeros([self.config['n_jk'], self.config['nbins_syst']])
        djk = len(sys_use) // self.config['n_jk']
        for j in range(self.config['n_jk']):
            jk_mask = np.ones(len(sys_use), dtype=bool)
            # Remove chunk
            jk_mask[j*djk:(j+1)*djk] = False
            means[j,:], _, _ = binned_statistic(sys_use[jk_mask],
                                                ng_use[jk_mask],
                                                statistic='mean',
                                                bins=bin_edges)
        err = np.std(means, axis=0) * np.sqrt(self.config['n_jk']-1.)

        return bin_centers, bin_centers*sys_mean, mean, err

    def parse_input(self) :
        """
        Check sanity of input parameters.
        """
        if self.config['sys_collapse_type']=='average':
            self.sys_map_offset=0
        elif self.config['sys_collapse_type']=='median':
            self.sys_map_offset=2
        else:
            raise ValueError('Systematic map flattening mode %s unknown. Use \'average\' or \'median\''%(self.config['sys_collapse_type']))
        self.xlabels = {'airmass' : 'Airmass',
                        'ccdtemp' : r'CCD Temperature [$^{\circ}$C]',
                        'ellipt' : 'PSF Ellipticity',
                        'exptime' : 'Exposure Time [s]',
                        'nvisit' : 'Number of visits',
                        'seeing' : 'Seeing [pixels]',
                        'sigma_sky' : 'Sky noise [ADU]',
                        'skylevel' : 'Sky level [ADU]',
                        'dust' : 'Extinction',
                        'stars' : 'Stars per pixel',
                        'depth' : '5-sigma depth'}
        #Bands
        self.bands=['g','r','i','z','y']
        print(self.get_output('systmap_plots'))
        os.system('mkdir -p ' + self.get_output('systmap_plots'))
        
    def get_mask(self):
        print("Reading mask")
        self.fsk,mp_depth=read_flat_map(self.get_input("depth_map"),i_map=0)
        mp_depth[np.isnan(mp_depth)]=0; mp_depth[mp_depth>40]=0
        msk_depth=np.zeros_like(mp_depth); msk_depth[mp_depth>=self.config['depth_cut']]=1

        fskb,self.mskfrac=read_flat_map(self.get_input("masked_fraction"),i_map=0)
        compare_infos(self.fsk,fskb)

        msk_bo=np.zeros_like(self.mskfrac); msk_bo[self.mskfrac>self.config['mask_thr']]=1
        self.msk_bi=msk_bo*msk_depth

    def get_sysmaps(self):
        print("Reading systematic maps")
        self.temps={}
        id_band=self.bands.index(self.config['band'])

        #Depth
        _,self.temps['depth']=read_flat_map(self.get_input("depth_map"),i_map=0)
        #Dust
        _,self.temps['dust']=read_flat_map(self.get_input("dust_map"),i_map=id_band)
        #Stars
        _,self.temps['stars']=read_flat_map(self.get_input("star_map"),i_map=0)
        for oc in ['ccdtemp','airmass','exptime','skylevel','sigma_sky','seeing','ellipt','nvisit']:
            for i_b,b in enumerate(self.bands):
                name=oc+'_'+b
                _,self.temps[name]=read_flat_map(self.get_input(oc+"_maps"),
                                                 i_map=i_b+5*self.sys_map_offset)

    def get_nmaps(self):
        hdul=fits.open(self.get_input('ngal_maps'))
        if len(hdul)%2!=0 :
            raise ValueError("Input file should have two HDUs per map")
        nbins=len(hdul)//2
        self.ng={}
        for ib in range(nbins):
            _,self.ng['bin%d'%(ib+1)]=read_flat_map(None,hdu=hdul[2*ib])

    def analyze_bin_systematic(self, bin_name, syst_name, bands=False):
        ng = self.ng[bin_name]

        if bands:
            maps = [self.temps[syst_name+'_'+b] for b in self.bands]
        else:
            maps = [self.temps[syst_name]]

        # Compute all statistics
        nmaps = len(maps)
        means = np.zeros([nmaps, self.config['nbins_syst']])
        errs = np.zeros([nmaps, self.config['nbins_syst']])
        bin_centers_resc = np.zeros([nmaps, self.config['nbins_syst']])
        bin_centers = np.zeros([nmaps, self.config['nbins_syst']])
        for imp, mp in enumerate(maps):
            aux_centers, aux_centers_resc, aux_mean, aux_err = self.compute_stats(ng, mp)
            means[imp, :] = aux_mean
            errs[imp, :] = aux_err
            bin_centers[imp, :] = aux_centers
            bin_centers_resc[imp, :] = aux_centers_resc

        # Save to file
        suffix = bin_name + "_" + syst_name
        prefix_save = os.path.join(self.get_output('systmap_plots'), suffix)
        np.savez(prefix_save, x=bin_centers_resc, x_rescaled=bin_centers,
                 mean=means, error=errs)

        # Get plot
        f,ax=plt.subplots(nmaps,1,figsize=(5,4*nmaps))
        if nmaps==1:
            ax=[ax]
        for i in range(nmaps):
            ax[i].errorbar(bin_centers[i], means[i], errs[i], fmt='o')
            ax[i].set_ylabel('$n/\\bar{n}$', fontsize=16)
            if nmaps!=1:
                ax[i].text(0.9, 0.9, self.bands[i], transform=ax[i].transAxes)
        ax[-1].set_xlabel(self.xlabels[syst_name], fontsize=16)
        f.tight_layout()
        f.savefig(prefix_save+'.pdf')
        plt.close(f)
        
    def run(self):
        """
        Main routine. This stage:
        - Creates number density maps from the reduced catalog for a set of redshift bins.
        - Calculates the associated N(z)s for each bin using different methods.
        - Stores the above into a single FITS file
        """
        self.parse_input()
        self.get_mask()

        self.get_sysmaps()
        print(self.temps.keys())

        self.get_nmaps()
        print(self.ng.keys())

        for bn in self.ng.keys():
            print(bn)
            for sn in ['depth','dust','stars']:
                print(" "+sn)
                self.analyze_bin_systematic(bn,sn,bands=False)
            for sn in ['ccdtemp','airmass','exptime','skylevel',
                       'sigma_sky','seeing','ellipt','nvisit']:
                print(" "+sn)
                self.analyze_bin_systematic(bn,sn,bands=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
