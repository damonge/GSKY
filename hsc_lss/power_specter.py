from ceci import PipelineStage
from .types import FitsFile,ASCIIFile,BinaryFile,NpzFile,SACCFile,DummyFile
import numpy as np
from .flatmaps import read_flat_map,compare_infos
from astropy.io import fits
import pymaster as nmt
from .tracer import Tracer
import os
import sacc
from scipy.interpolate import interp1d

class PowerSpecter(PipelineStage) :
    name="PowerSpecter"
    inputs=[('masked_fraction',FitsFile),('ngal_maps',FitsFile),
            ('dust_map',FitsFile),('star_map',FitsFile),('depth_map',FitsFile),
            ('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),('exptime_maps',FitsFile),
            ('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),('seeing_maps',FitsFile),
            ('ellipt_maps',FitsFile),('nvisit_maps',FitsFile),('cosmos_weights',FitsFile),
            ('syst_masking_file',ASCIIFile)]
    outputs=[('dummy',DummyFile)]
    config_options={'ell_bpws':[100.0,200.0,300.0,
                                400.0,600.0,800.0,
                                1000.0,1400.0,1800.0,
                                2200.0,3000.0,3800.0,
                                4600.0,6200.0,7800.0,
                                9400.0,12600.0,15800.0],
                    'oc_dpj_list': ['airmass','seeing','sigma_sky'],
                    'depth_cut':24.5,'band':'i','mask_thr':0.5,'guess_spectrum':'NONE',
                    'gaus_covar_type':'analytic','oc_all_bands':True,
                    'mask_systematics':False,'noise_bias_type':'analytic',
                    'output_run_dir':None,'sys_collapse_type':'average'}

    def read_map_bands(self,fname,read_bands,bandname,offset=0) :
        """
        Reads maps from file.
        :param fname: file name
        :param read_bands: if True, read map in all bands
        :param bandname: if `read_bands==False`, then read only the map for this band.
        """
        if read_bands :
            temp=[]
            for i in range(5):
                i_map=i+5*offset
                fskb,t=read_flat_map(fname,i_map=i_map)
                compare_infos(self.fsk,fskb)
                temp.append(t)
        else :
            i_map=['g','r','i','z','y'].index(bandname)+5*offset
            fskb,temp=read_flat_map(fname,i_map=i_map)
            compare_infos(self.fsk,fskb)
            temp=[temp]

        return temp

    def get_sacc_windows(self,wsp) :
        """
        Get window functions for each bandpower so they can be stored into the final SACC files.
        """
        #Compute window functions
        nbands=wsp.wsp.bin.n_bands
        l_arr=np.arange(self.lmax+1)
        if not os.path.isfile(self.get_output_fname('windows_l',ext='npz')) :
            print("Computing window functions")
            windows=np.zeros([nbands,self.lmax+1])
            t_hat=np.zeros(self.lmax+1);
            for il,l in enumerate(l_arr) :
                t_hat[il]=1.;
                windows[:,il]=wsp.decouple_cell(wsp.couple_cell(l_arr,[t_hat]))
                t_hat[il]=0.;
            np.savez(self.get_output_fname('windows_l'),windows=windows)
        else :
            print("Reading window functions")
            windows=np.load(self.get_output_fname('windows_l',ext='npz'))['windows']

        windows_sacc=[]
        #i_x=0
        for i in range(self.nbins) :
            for j in range(i,self.nbins) :
                for b in range(nbands) :
                    windows_sacc.append(sacc.Window(l_arr,windows[b]))

        return windows_sacc

    def get_noise(self,tracers,wsp,bpws,nsims=1000) :
        """
        Get an estimate of the noise bias.
        :param tracers: list of Tracers.
        :param wsp: NaMaster workspace.
        :param bpws: NaMaster bandpowers.
        :param nsims: number of simulations to use (if using them).
        """
        if self.config['noise_bias_type']=='analytic' :
            return self.get_noise_analytic(tracers,wsp)
        elif self.config['noise_bias_type']=='pois_sim' :
            return self.get_noise_simulated(tracers,wsp,bpws,nsims)

    def get_noise_analytic(self,tracers,wsp) :
        """
        Get an analytical estimate of the noise bias.
        :param tracers: list of Tracers.
        :param wsp: NaMaster workspace.
        """
        nls_all=np.zeros([self.ncross,self.nell])
        i_x=0
        for i in range(self.nbins) :
            for j in range(i,self.nbins) :
                if i==j : #Add shot noise in auto-correlation
                    t=tracers[i]
                    corrfac=np.sum(t.weight)/(t.fsk.nx*t.fsk.ny)
                    nl=np.ones(self.nell)*corrfac/t.ndens_perad
                    nls_all[i_x]=wsp.decouple_cell([nl])[0]
                i_x+=1
        return nls_all
        
    def get_noise_simulated(self,tracers,wsp,bpws,nsims) :
        """
        Get a simulated estimate of the noise bias.
        :param tracers: list of Tracers.
        :param wsp: NaMaster workspace.
        :param bpws: NaMaster bandpowers.
        :param nsims: number of simulations to use (if using them).
        """
        def randomize_deltag_map(tracer,seed) :
            """
            Creates a randomised version of the input map map by assigning the
            galaxies in the surevy to random pixels in the map. Basically it rotates each
            galaxy by a random angle but not rotating it out of the survey footprint.
            :param map: masked galaxy overdensity map which needs to randomised
            :param Ngal: number of galaxies used to create the map
            :return randomised_map: a randomised version of the masked input map
            """
            
            mask = tracer.weight.reshape([tracer.fsk.ny, tracer.fsk.nx])
            Ngal = int(tracer.Ngal)

            np.random.seed(seed=seed)
            maskpixy,maskpixx=np.where(mask!=0.)
            galpix_mask=np.random.choice(np.arange(maskpixx.shape[0]),size=Ngal,
                                         p=mask[mask != 0.]/np.sum(mask[mask != 0.]))
            galpixx=maskpixx[galpix_mask]
            galpixy=maskpixy[galpix_mask]

            maskshape=mask.shape
            ny,nx=maskshape
            ipix=galpixx+nx*galpixy

            randomized_nmap=np.bincount(ipix,minlength=nx*ny)

            randomized_deltamap=np.zeros_like(randomized_nmap,dtype='float')
            ndens=np.sum(randomized_nmap*tracer.mask_binary)/np.sum(tracer.weight)
            randomized_deltamap[tracer.goodpix]=randomized_nmap[tracer.goodpix]/(ndens*tracer.masked_fraction[tracer.goodpix])-1
            randomized_deltamap=randomized_deltamap.reshape(maskshape)

            return randomized_deltamap

        nls_all=np.zeros([self.ncross,self.nell])
        i_x=0
        for i in range(self.nbins) :
            for j in range(i,self.nbins) :
                if i==j: #Add shot noise in the auto-correlation
                    tracer=tracers[i]
                    mask=tracer.weight.reshape([tracer.fsk.ny,tracer.fsk.nx])
                    ncl_uncoupled=np.zeros((nsims,self.nell))
                    for ii in range(nsims) :
                        randomized_map=randomize_deltag_map(tracer,ii+nsims*i)
                        f0=nmt.NmtFieldFlat(np.radians(self.fsk.lx),np.radians(self.fsk.ly),mask,
                                            [randomized_map])
                        ncl_uncoupled[ii,:]=wsp.decouple_cell(nmt.compute_coupled_cell_flat(f0,f0,bpws))
                    nls_all[i_x]=np.mean(ncl_uncoupled,axis=0)
                i_x+=1

        return nls_all

    def get_dpj_bias(self,trc,lth,clth,cl_coupled,wsp,bpws) :
        """
        Estimate the deprojection bias
        :param trc: list of Tracers.
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param cl_coupled: mode-coupled measurements of the power spectrum (before subtracting the deprojection bias).
        :param wsp: NaMaster workspace.
        :param bpws: NaMaster bandpowers.
        """
        #Compute deprojection bias
        if os.path.isfile(self.get_output_fname('dpj_bias',ext='sacc')) :
            print("Reading deprojection bias")
            s=sacc.SACC.loadFromHDF(self.get_output_fname('dpj_bias',ext='sacc'))
            cls_deproj_all=s.mean.vector.reshape([self.ncross,self.nell])
        else :
            print("Computing deprojection bias")
            cls_deproj_all=[]
            i_x=0
            for i in range(self.nbins) :
                for j in range(i,self.nbins) :
                    print(i,j)
                    cl_deproj_bias=nmt.deprojection_bias_flat(trc[i].field,trc[j].field,bpws,
                                                              lth,[clth[i_x]])[0]
                    cls_deproj_all.append(cl_deproj_bias)
                    i_x+=1
            cls_deproj_all=np.array(cls_deproj_all)

        #Remove deprojection bias
        cls_all=[]
        i_x=0
        for i in range(self.nbins) :
            for j in range(i,self.nbins) :
                cls_all.append(wsp.decouple_cell([cl_coupled[i_x]],
                                                 cl_bias=[cls_deproj_all[i_x]])[0])
                i_x+=1
        return np.array(cls_all),cls_deproj_all

    def get_cl_guess(self,ld,cld) :
        """
        Read or compute the guess power spectra.
        :param ld: list of multipoles at which the data power spectra have been measured.
        :param cld: list of power spectrum measurements from the data.
        """
        lth=np.arange(2,self.lmax+1)
        clth=np.zeros([self.ncross,len(lth)])
        if self.config['guess_spectrum']=='NONE' :
            print("Interpolating data power spectra")
            l_use=ld
            cl_use=cld
        else:
            data=np.loadtxt(self.config['guess_spectrum'],unpack=True)
            l_use=data[0]
            cl_use=data[1:]
            if len(clth)!=self.ncross :
                raise ValueError("Theory power spectra have a wrong shape")
        #Interpolate
        for i in range(self.ncross) :
            clf=interp1d(l_use,cl_use[i],bounds_error=False,fill_value=0,kind='linear')
            clth[i,:]=clf(lth)
            clth[i,lth<=l_use[0]]=cl_use[i,0]
            clth[i,lth>=l_use[-1]]=cl_use[i,-1]

        return lth,clth

    def get_power_spectra(self,trc,wsp,bpws) :
        """
        Compute all possible power spectra between pairs of tracers
        :param trc: list of Tracers.
        :param wsp: NaMaster workspace.
        :param bpws: NaMaster bandpowers.
        """
        cls_all=[]
        cls_coupled=[]
        for i in range(self.nbins) :
            for j in range(i,self.nbins) :
                cl_coupled=nmt.compute_coupled_cell_flat(trc[i].field,trc[j].field,bpws)
                cls_all.append(wsp.decouple_cell(cl_coupled)[0])
                cls_coupled.append(cl_coupled[0])
        return np.array(cls_all),np.array(cls_coupled)

    def get_covar(self,lth,clth,bpws,tracers,wsp,temps,cl_dpj_all) :
        """
        Estimate the power spectrum covariance
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param bpws: NaMaster bandpowers.
        :params tracers: tracers.
        :param wsp: NaMaster workspace.
        :param temps: list of contaminant templates.
        :params cl_dpj_all: list of deprojection biases for each bin pair combination.
        """
        if self.config['gaus_covar_type']=='analytic' :
            print("Computing analytical Gaussian covariance")
            cov=self.get_covar_analytic(lth,clth,bpws,tracers,wsp)
        elif self.config['gaus_covar_type']=='gaus_sim' :
            print("Computing simulated Gaussian covariance")
            cov=self.get_covar_gaussim(lth,clth,bpws,wsp,temps,cl_dpj_all)

        return cov
            
    def get_mcm(self,tracers,bpws) :
        """
        Get NmtWorkspaceFlat for our mask
        """
        wsp=nmt.NmtWorkspaceFlat()
        if not os.path.isfile(self.get_output_fname('mcm',ext='dat')) :
            print("Computing MCM")
            wsp.compute_coupling_matrix(tracers[0].field,tracers[0].field,bpws)
            wsp.write_to(self.get_output_fname('mcm',ext='dat'))
        else :
            print("Reading MCM")
            wsp.read_from(self.get_output_fname('mcm',ext='dat'))
        
        return wsp

    def get_covar_mcm(self,tracers,bpws):
        """
        Get NmtCovarianceWorkspaceFlat for our mask
        """
        cwsp=nmt.NmtCovarianceWorkspaceFlat()
        if not os.path.isfile(self.get_output_fname('cov_mcm',ext='dat')) :
            print("Computing covariance MCM")
            cwsp.compute_coupling_coefficients(tracers[0].field,
                                               tracers[0].field,bpws)
            cwsp.write_to(self.get_output_fname('cov_mcm',ext='dat'))
        else :
            print("Reading covariance MCM")
            cwsp.read_from(self.get_output_fname('cov_mcm',ext='dat'))
        
        return cwsp

    def get_covar_gaussim(self,lth,clth,bpws,wsp,temps,cl_dpj_all) :
        """
        Estimate the power spectrum covariance from Gaussian simulations
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param bpws: NaMaster bandpowers.
        :param wsp: NaMaster workspace.
        :param temps: list of contaminatn templates.
        :params cl_dpj_all: list of deprojection biases for each bin pair combination.
        """
        #Create a dummy file for the covariance MCM
        f=open(self.get_output_fname('cov_mcm',ext='dat'),"w")
        f.close()

        #Setup
        nsims=10*self.ncross*self.nell
        print("Computing covariance from %d Gaussian simulations"%nsims)
        msk_binary=self.msk_bi.reshape([self.fsk.ny,self.fsk.nx])
        weights=(self.msk_bi*self.mskfrac).reshape([self.fsk.ny,self.fsk.nx])
        if temps is not None :
            conts=[[t.reshape([self.fsk.ny,self.fsk.nx])] for t in temps]
            cl_dpj=[[c] for c in cl_dpj_all]
        else :
            conts=None
            cl_dpj=[None for i in range(self.ncross)]

        #Iterate
        cells_sims=[]
        for isim in range(nsims) :
            if isim%100==0 :
                print(" %d-th sim"%isim)
            #Generate random maps
            mps=nmt.synfast_flat(self.fsk.nx,self.fsk.ny,
                                 np.radians(self.fsk.lx),np.radians(self.fsk.ly),
                                 clth,np.zeros(self.nbins),seed=1000+isim)
            #Nmt fields
            flds=[nmt.NmtFieldFlat(np.radians(self.fsk.lx),np.radians(self.fsk.ly),weights,
                                   [m],templates=conts) for m in mps]
            #Compute power spectra (possibly with deprojection)
            i_x=0
            cells_this=[]
            for i in range(self.nbins) :
                for j in range(i,self.nbins) :
                    cl=nmt.compute_coupled_cell_flat(flds[i],flds[j],bpws)
                    cells_this.append(wsp.decouple_cell(cl,cl_bias=cl_dpj[i_x])[0])
                    i_x+=1
            cells_sims.append(np.array(cells_this).flatten())
        cells_sims=np.array(cells_sims)
        #Save simulations for further 
        np.savez(self.get_output_fname('gaucov_sims'),cl_sims=cells_sims)
        
        #Compute covariance
        covar=np.cov(cells_sims.T)
        return covar

    def get_covar_analytic(self,lth,clth,bpws,tracers,wsp) :
        """
        Estimate the power spectrum covariance analytically
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param bpws: NaMaster bandpowers.
        :param tracers: tracers.
        :param wsp: NaMaster workspace.
        """
        #Create a dummy file for the covariance MCM
        f=open(self.get_output_fname('gaucov_sims',ext='npz'),"w")
        f.close()

        covar=np.zeros([self.ncross*self.nell,self.ncross*self.nell])
        cwsp=self.get_covar_mcm(tracers,bpws)

        ix_1=0
        for i1 in range(self.nbins) :
            for j1 in range(i1,self.nbins) :
                ix_2=0
                for i2 in range(self.nbins) :
                    for j2 in range(i2,self.nbins) :
                        ca1b1=clth[self.ordering[i1,i2]]
                        ca1b2=clth[self.ordering[i1,j2]]
                        ca2b1=clth[self.ordering[j1,i2]]
                        ca2b2=clth[self.ordering[j1,j2]]
                        cov_here=nmt.gaussian_covariance_flat(cwsp,0,0,0,0,
                                                              lth,[ca1b1],[ca1b2],[ca2b1],[ca2b2],wsp)
                        covar[ix_1*self.nell:(ix_1+1)*self.nell,:][:,ix_2*self.nell:(ix_2+1)*self.nell]=cov_here
                        ix_2+=1
                ix_1+=1

        return covar

    def get_masks(self) :
        """
        Read or compute all binary masks and the masked fraction map.
        """
        #Depth-based mask
        self.fsk,mp_depth=read_flat_map(self.get_input("depth_map"),i_map=0)
        mp_depth[np.isnan(mp_depth)]=0; mp_depth[mp_depth>40]=0
        msk_depth=np.zeros_like(mp_depth); msk_depth[mp_depth>=self.config['depth_cut']]=1

        fskb,mskfrac=read_flat_map(self.get_input("masked_fraction"),i_map=0)
        compare_infos(self.fsk,fskb)
        
        #Create binary mask (fraction>threshold and depth req.)
        msk_bo=np.zeros_like(mskfrac); msk_bo[mskfrac>self.config['mask_thr']]=1
        msk_bi=msk_bo*msk_depth

        if self.config['mask_systematics'] :
            #Mask systematics
            msk_syst=msk_bi.copy()
            #Read systematics cut data
            data_syst=np.genfromtxt(self.get_input('syst_masking_file'),
                                    dtype=[('name','|U32'),('band','|U4'),('gl','|U4'),('thr','<f8')])
            for d in data_syst :
                #Read systematic
                if d['name'].startswith('oc_'):
                    sysmap=self.read_map_bands(self.get_input(d['name'][3:]+'_maps'),False,d['band'],
                                               offset=self.sys_map_offset)[0]
                elif d['name']=='dust':
                    sysmap=self.read_map_bands(self.get_input('dust_map'),False,d['band'])[0]
                else :
                    raise KeyError("Unknown systematic name "+d['name'])
    
                #Divide by mean
                sysmean=np.sum(msk_bi*mskfrac*sysmap)/np.sum(msk_bi*mskfrac)
                sysmap/=sysmean

                #Apply threshold
                msk_sys_this=msk_bi.copy(); fsky_pre=np.sum(msk_syst)
                if d['gl']=='<' :
                    msk_sys_this[sysmap<d['thr']]=0
                else :
                    msk_sys_this[sysmap>d['thr']]=0
                msk_syst*=msk_sys_this
                fsky_post=np.sum(msk_syst)
                print(' '+d['name']+d['gl']+'%.3lf'%(d['thr'])+
                      ' removes ~%.2lf per-cent of the available sky'%((1-fsky_post/fsky_pre)*100))
            print(' All systematics remove %.2lf per-cent of the sky'%((1-np.sum(msk_syst)/np.sum(msk_bi))*100))
            self.fsk.write_flat_map(self.get_output_fname("mask_syst",ext="fits"),msk_syst)

            msk_bi*=msk_syst

        return msk_bi,mskfrac,mp_depth

    def get_tracers(self,temps) :
        """
        Produce a Tracer for each redshift bin. Do so with and without contaminant deprojection.
        :param temps: list of contaminant tracers
        """
        hdul=fits.open(self.get_input('ngal_maps'))
        if len(hdul)%2!=0 :
            raise ValueError("Input file should have two HDUs per map")
        nbins=len(hdul)//2
        tracers_nocont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None)
                        for i in range(nbins)]
        tracers_wcont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=temps)
                       for i in range(nbins)]
        hdul.close()
        return tracers_nocont,tracers_wcont

    def get_contaminants(self) :
        """
        Read all contaminant maps.
        """
        temps=[]
        #Depth
        temps.append(self.mp_depth)
        #Dust
        for t in self.read_map_bands(self.get_input('dust_map'),False,self.config['band']) :
            temps.append(t)
        #Stars
        fskb,t=read_flat_map(self.get_input('star_map'),i_map=0)
        compare_infos(self.fsk,fskb)
        temps.append(t)
        #Observing conditions
        for oc in self.config['oc_dpj_list'] :
            for t in self.read_map_bands(self.get_input(oc+'_maps'),
                                         self.config['oc_all_bands'],
                                         self.config['band'],offset=self.sys_map_offset) :
                temps.append(t)
        temps=np.array(temps)
        #Remove mean
        for i_t,t in enumerate(temps) :
            temps[i_t]-=np.sum(self.msk_bi*self.mskfrac*t)/np.sum(self.msk_bi*self.mskfrac)

        return temps

    def get_output_fname(self,name,ext=None):
        fname=self.output_dir+name
        if ext is not None:
            fname+='.'+ext
        return fname

    def parse_input(self) :
        """
        Check sanity of input parameters.
        """
        # This is a hack to get the path of the root output directory.
        # It should be easy to get this from ceci, but I don't know how to.
        self.output_dir=self.get_output('dummy',final_name=True)[:-5]
        if self.config['output_run_dir'] is not None:
            self.output_dir+=self.config['output_run_dir']+'/'
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if (self.config['noise_bias_type']!='analytic') and (self.config['noise_bias_type']!='pois_sim') :
            raise ValueError('Noise bias calculation must be either \'analytic\' or \'pois_sim\'')
        if (self.config['gaus_covar_type']!='analytic') and (self.config['gaus_covar_type']!='gaus_sim') :
            raise ValueError('Gaussian covariance calculation must be either \'analytic\' or \'pois_sim\'')
        if self.config['guess_spectrum']!='NONE' :
            if not os.path.isfile(self.config['guess_spectrum']) :
                raise ValueError('Guess spectrum must be either \'NONE\' or an existing ASCII file')
        if self.config['sys_collapse_type']=='average':
            self.sys_map_offset=0
        elif self.config['sys_collapse_type']=='median':
            self.sys_map_offset=2
        else:
            raise ValueError('Systematic map flattening mode %s unknown. Use \'average\' or \'median\''%(self.config['sys_collapse_type']))

        return

    def get_sacc_tracers(self,tracers) :
        """
        Generate a list of SACC tracers from the input Tracers.
        """
        sacc_tracers=[]
        for i_t,t in enumerate(tracers) :
            z=(t.nz_data['z_i']+t.nz_data['z_f'])*0.5
            nz=t.nz_data['nz_cosmos']
            T=sacc.Tracer('bin_%d'%i_t,'point',z,nz,exp_sample="HSC_DESC")
            T.addColumns({'nz_'+c:t.nz_data['nz_'+c] 
                          for c in ['demp','ephor','ephor_ab','frankenz','nnpz']})
            sacc_tracers.append(T)

        return sacc_tracers

    def write_vector_to_sacc(self,fname_out,sacc_t,sacc_b,cls,covar=None,verbose=False) :
        """
        Write a vector of power spectrum measurements into a SACC file.
        :param fname_out: path to output file
        :param sacc_t: list of SACC tracers
        :param sacc_b: SACC Binning object.
        :param cls: list of power spectrum measurements.
        :param covar: covariance matrix:
        :param verbose: do you want to print out information about the SACC file?
        """
        sacc_mean=sacc.MeanVec(cls.flatten())
        if covar is None :
            sacc_precision=None
        else :
            sacc_precision=sacc.Precision(covar,"dense",is_covariance=True, binning=sacc_b)

        sacc_meta={'Area_rad':self.area_patch}
        s=sacc.SACC(sacc_t,sacc_b,sacc_mean,precision=sacc_precision,meta=sacc_meta)
        if verbose :
            s.printInfo()
        s.saveToHDF(fname_out)

    def get_sacc_binning(self,ell_eff,lini,lend,windows=None) :
        """
        Generate a SACC binning object.
        :param ell_eff: list of effective multipoles.
        :param lini,lend: bandpower edges.
        :param windows: optional list of bandpower window functions.
        """
        typ,ell,dell,t1,q1,t2,q2=[],[],[],[],[],[],[]
        for t1i in range(self.nbins) :
            for t2i in range(t1i,self.nbins) :
                for i_l,l in enumerate(ell_eff) :
                    typ.append('F')
                
                    ell.append(l)
                    dell.append(lend[i_l]-lini[i_l])
                    t1.append(t1i)
                    q1.append('P')
                    t2.append(t2i)
                    q2.append('P')

        if windows is None :
            return sacc.Binning(typ,ell,t1,q1,t2,q2,deltaLS=dell)
        else :
            return sacc.Binning(typ,ell,t1,q1,t2,q2,deltaLS=dell,windows=windows)
                
    def run(self) :
        """
        Main function.
        This stage:
        - Produces measurements of the power spectrum with and without contaminant deprojections.
        - Estimates the noise bias
        - Estimates the covariance matrix
        - Estimates the deprojection bias
        """
        self.parse_input()

        print("Reading mask")
        self.msk_bi,self.mskfrac,self.mp_depth=self.get_masks()

        print("Computing area")
        self.area_pix=np.radians(self.fsk.dx)*np.radians(self.fsk.dy)
        self.area_patch=np.sum(self.msk_bi*self.mskfrac)*self.area_pix
        self.lmax=int(180.*np.sqrt(1./self.fsk.dx**2+1./self.fsk.dy**2))

        print("Reading contaminants")
        temps=self.get_contaminants()

        print("Setting bandpowers")
        lini=np.array(self.config['ell_bpws'])[:-1]
        lend=np.array(self.config['ell_bpws'])[ 1:]
        bpws=nmt.NmtBinFlat(lini,lend)
        ell_eff=bpws.get_effective_ells()

        print("Generating tracers")
        tracers_nc,tracers_wc=self.get_tracers(temps)
        self.nbins=len(tracers_nc)

        print("Translating into SACC tracers")
        tracers_sacc=self.get_sacc_tracers(tracers_nc)

        self.ordering=np.zeros([self.nbins,self.nbins],dtype=int)
        ix=0
        for i in range(self.nbins) :
            for j in range(i,self.nbins) :
                self.ordering[i,j]=ix
                if j!=i :
                    self.ordering[j,i]=ix
                ix+=1

        print("Getting MCM")
        wsp=self.get_mcm(tracers_nc,bpws)

        print("Computing window function")
        windows=self.get_sacc_windows(wsp)

        print("Computing SACC binning")
        #No windows
        binning_nw=self.get_sacc_binning(ell_eff,lini,lend,windows=None)
        #With windows
        binning_ww=self.get_sacc_binning(ell_eff,lini,lend,windows=windows)

        print("Computing power spectra")
        print(" No deprojections")
        cls_wodpj,_=self.get_power_spectra(tracers_nc,wsp,bpws)
        print(" W. deprojections")
        cls_wdpj,cls_wdpj_coupled=self.get_power_spectra(tracers_wc,wsp,bpws)
        self.ncross,self.nell=cls_wodpj.shape

        print("Getting guess power spectra")
        lth,clth=self.get_cl_guess(ell_eff,cls_wdpj)

        print("Computing deprojection bias")
        cls_wdpj,cls_deproj=self.get_dpj_bias(tracers_wc,lth,clth,cls_wdpj_coupled,wsp,bpws)

        print("Computing covariance")
        cov_wodpj=self.get_covar(lth,clth,bpws,tracers_wc,wsp,None,None)
        if self.config['gaus_covar_type']=='analytic' :
            cov_wdpj=cov_wodpj.copy()
        else :
            cov_wdpj=self.get_covar(lth,clth,bpws,tracers_wc,wsp,temps,cls_deproj)

        print("Computing noise bias")
        nls=self.get_noise(tracers_nc,wsp,bpws)

        print("Writing output")
        print(self.get_output_fname('noi_bias',ext='sacc'))
        self.write_vector_to_sacc(self.get_output_fname('noi_bias',ext='sacc'),tracers_sacc,
                                  binning_nw,nls,verbose=False)
        self.write_vector_to_sacc(self.get_output_fname('dpj_bias',ext='sacc'),tracers_sacc,
                                  binning_nw,cls_deproj,verbose=False)
        self.write_vector_to_sacc(self.get_output_fname('power_spectra_wodpj',ext='sacc'),tracers_sacc,
                                  binning_ww,cls_wodpj,covar=cov_wodpj,verbose=False)
        self.write_vector_to_sacc(self.get_output_fname('power_spectra_wdpj',ext='sacc'),tracers_sacc,
                                  binning_ww,cls_wdpj,covar=cov_wdpj,verbose=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
