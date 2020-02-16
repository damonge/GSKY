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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO: Names of files to read
#TODO: COSMOS nz for shear weights

class PowerSpecter(PipelineStage) :
    name="PowerSpecter"
    inputs=[('masked_fraction',FitsFile),('ngal_maps',FitsFile),('shear_maps',FitsFile),
            ('dust_map',FitsFile),('star_map',FitsFile),('depth_map',FitsFile),
            ('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),('exptime_maps',FitsFile),
            ('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),('seeing_maps',FitsFile),
            ('ellipt_maps',FitsFile),('nvisit_maps',FitsFile),('cosmos_weights',FitsFile),
            ('syst_masking_file',ASCIIFile)]
    outputs=[('dummy',DummyFile), ('mcm', BinaryFile)]
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

    def get_windows(self,wsp) :
        """
        Get window functions for each bandpower so they can be stored into the final SACC files.
        """

        # Compute window functions
        logger.info("Computing window functions.")
        nbands = wsp[0, 0].wsp.bin.n_bands
        l_arr = np.arange(self.lmax + 1)

        windows_list = [[0 for i in range(self.ntracers)] for ii in range(self.ntracers)]

        if self.get_input('ngal_maps') != 'NONE':
            logger.info('Number density maps provided.')
            if not os.path.isfile(self.get_output_fname('windows_l', ext='npz')[0]):
                logger.info("Computing window functions for counts.")
                windows_counts = np.zeros([nbands, self.lmax + 1])
                t_hat = np.zeros(self.lmax + 1)
                for il, l in enumerate(l_arr):
                    t_hat[il] = 1.
                    windows_counts[:, il] = wsp[0, 0].decouple_cell(wsp[0, 0].couple_cell(l_arr, [t_hat]))
                    t_hat[il] = 0.
                np.savez(self.get_output_fname('windows_l')[0], windows=windows_counts)
            else:
                logger.info("Reading window functions for counts.")
                windows_counts = np.load(self.get_output_fname('windows_l', ext='npz')[0])['windows']

            if self.get_input('shear_maps') != 'NONE':
                logger.info('Number density and shear maps provided.')
                for i in range(self.ntracers):
                    for ii in range(i, self.ntracers):
                        if i < self.ncounts_maps and ii < self.ncounts_maps:
                            windows_list[i, ii] = windows_counts
                        elif i == 0 and ii >= self.ncounts_maps:
                            if not os.path.isfile(self.get_output_fname('windows_l', ext='npz')[self.ordering[i, ii]]):
                                logger.info("Computing window functions for counts x shear.")
                                windows = np.zeros([nbands, self.lmax + 1])
                                t_hat = np.zeros(self.lmax + 1)
                                for il, l in enumerate(l_arr):
                                    t_hat[il] = 1.
                                    windows[:, il] = wsp[i, ii].decouple_cell(wsp[i, ii].couple_cell(l_arr, [t_hat]))
                                    t_hat[il] = 0.
                                np.savez(self.get_output_fname('windows_l')[0], windows=windows)
                            else:
                                logger.info("Reading window functions for counts x shear.")
                                windows = np.load(self.get_output_fname('windows_l', ext='npz')[self.ordering[i, ii]])['windows']

                            windows_list[i, ii] = windows
                        elif i != 0 and i < self.ncounts_maps and ii >= self.ncounts_maps:
                            windows_list[i, ii] = windows_list[0, ii]
                        elif i >= self.ncounts_maps and ii >= self.ncounts_maps:
                            if not os.path.isfile(self.get_output_fname('windows_l', ext='npz')[self.ordering[i, ii]]):
                                logger.info("Computing window functions for shear.")
                                windows = np.zeros([nbands, self.lmax + 1])
                                t_hat = np.zeros(self.lmax + 1)
                                for il, l in enumerate(l_arr):
                                    t_hat[il] = 1.
                                    windows[:, il] = wsp[i, ii].decouple_cell(wsp[i, ii].couple_cell(l_arr, [t_hat]))
                                    t_hat[il] = 0.
                                np.savez(self.get_output_fname('windows_l')[0], windows=windows)
                            else:
                                logger.info("Reading window functions for shear.")
                                windows = np.load(self.get_output_fname('windows_l', ext='npz')[self.ordering[i, ii]])['windows']

                            windows_list[i, ii] = windows
                        else:
                            raise RuntimeError("Messed-up indexing in window function computation.")

        # Only shear maps case
        else:
            logger.info('Shear maps provided.')
            for i in range(self.ntracers):
                for ii in range(i, self.ntracers):
                    if not os.path.isfile(self.get_output_fname('windows_l', ext='npz')[self.ordering[i, ii]]):
                        logger.info("Computing window functions for shear.")
                        windows = np.zeros([nbands, self.lmax + 1])
                        t_hat = np.zeros(self.lmax + 1)
                        for il, l in enumerate(l_arr):
                            t_hat[il] = 1.
                            windows[:, il] = wsp[i, ii].decouple_cell(wsp[i, ii].couple_cell(l_arr, [t_hat]))
                            t_hat[il] = 0.
                        np.savez(self.get_output_fname('windows_l')[0], windows=windows)
                    else:
                        logger.info("Reading window functions for shear.")
                        windows = np.load(self.get_output_fname('windows_l', ext='npz')[self.ordering[i, ii]])['windows']

                    windows_list[i, ii] = windows

        return windows_list

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
        for i in range(self.ntracers) :
            for j in range(i,self.ntracers) :
                if i==j : #Add shot noise in auto-correlation
                    tr_i, tr_j = self.pss2tracers(i, j)
                    t = tracers[tr_i]
                    if t.spin == 0:
                        corrfac=np.sum(t.weight)/(t.fsk.nx*t.fsk.ny)
                        nl=np.ones(self.nell)*corrfac/t.ndens_perad
                        nls_all[i_x]=wsp[tr_i, tr_j].decouple_cell([nl])[0]
                    elif t.spin == 2:
                        corrfac=np.sum(t.weight)/(t.fsk.nx*t.fsk.ny)
                        nl=np.mean(self.e1_2rms_cat+self.e1_2rms_cat)*corrfac/t.ndens_perad
                        nls_all[i_x]=wsp[tr_i, tr_j].decouple_cell([nl])[0]
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
            # Compute and remove deprojection bias
            map_i = 0
            map_j = 0
            for tr_i in range(self.ntracers):
                for tr_j in range(tr_i, self.ntracers):
                    if trc[tr_i].spin == 0 and trc[tr_j].spin == 0:
                        cl_deproj_bias_temp = nmt.compute_coupled_cell_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                                            lth, [clth[map_i, map_j]])
                        cl_deproj_temp = wsp.decouple_cell([cl_coupled[map_i, map_j]], cl_bias=cl_deproj_bias_temp)
                        cl_deproj_bias[map_i, map_j] = cl_deproj_bias_temp[0]
                        cl_deproj[map_i, map_j] = cl_deproj_temp[0]
                        map_j += 1
                    elif trc[tr_i].spin == 0 and tr[tr_j].spin == 2:
                        cl_deproj_bias_temp = nmt.compute_coupled_cell_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                                lth, [clth[map_i, map_j], clth[map_i, map_j + 1]])
                        cl_deproj_temp = wsp.decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i, map_j + 1]],
                                                           cl_bias=cl_deproj_bias_temp)
                        # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                        cl_deproj_bias_tempe = cl_deproj_bias_temp[0]
                        cl_deproj_bias_tempb = cl_deproj_bias_temp[1]
                        cl_deproj_bias[map_i, map_j] = cl_deproj_bias_tempe
                        cl_deproj_bias[map_i, map_j + 1] = cl_deproj_bias_tempb
                        cl_deproj_tempe = cl_deproj_temp[0]
                        cl_deproj_tempb = cl_deproj_temp[1]
                        cl_deproj[map_i, map_j] = cl_deproj_tempe
                        cl_deproj[map_i, map_j + 1] = cl_deproj_tempb
                        map_j += 2
                    elif trc[tr_i].spin == 2 and tr[tr_j].spin == 0:
                        cl_deproj_bias_temp = nmt.compute_coupled_cell_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                                lth, [clth[map_i, map_j], clth[map_i + 1, map_j]])
                        cl_deproj_temp = wsp.decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i + 1, map_j]],
                                                           cl_bias=cl_deproj_bias_temp)
                        # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                        cl_deproj_bias_tempe = cl_deproj_bias_temp[0]
                        cl_deproj_bias_tempb = cl_deproj_bias_temp[1]
                        cl_deproj_bias[map_i, map_j] = cl_deproj_bias_tempe
                        cl_deproj_bias[map_i + 1, map_j] = cl_deproj_bias_tempb
                        cl_deproj_tempe = cl_deproj_temp[0]
                        cl_deproj_tempb = cl_deproj_temp[1]
                        cl_deproj[map_i, map_j] = cl_deproj_tempe
                        cl_deproj[map_i + 1, map_j] = cl_deproj_tempb
                        map_j += 1
                    else:
                        cl_deproj_bias_temp = nmt.compute_coupled_cell_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                        lth, [clth[map_i, map_j], clth[map_i, map_j + 1],
                                                              clth[map_i + 1, map_j], clth[map_i + 1, map_j + 1]])
                        cl_deproj_temp = wsp.decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i, map_j + 1],
                                                            cl_coupled[map_i + 1, map_j], cl_coupled[map_i + 1, map_j + 1]],
                                                           cl_bias=cl_deproj_bias_temp)
                        # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
                        cl_deproj_bias_tempe = cl_deproj_bias_temp[0]
                        cl_deproj_bias_tempeb = cl_deproj_bias_temp[1]
                        cl_deproj_bias_tempbe = cl_deproj_bias_temp[2]
                        cl_deproj_bias_tempb = cl_deproj_bias_temp[3]
                        cl_deproj_bias[map_i, map_j] = cl_deproj_bias_tempe
                        cl_deproj_bias[map_i, map_j + 1] = cl_deproj_bias_tempeb
                        cl_deproj_bias[map_i + 1, map_j] = cl_deproj_bias_tempbe
                        cl_deproj_bias[map_i + 1, map_j + 1] = cl_deproj_bias_tempb
                        cl_deproj_tempe = cl_deproj_temp[0]
                        cl_deproj_tempeb = cl_deproj_temp[1]
                        cl_deproj_tempbe = cl_deproj_temp[2]
                        cl_deproj_tempb = cl_deproj_temp[3]
                        cl_deproj[map_i, map_j] = cl_deproj_tempe
                        cl_deproj[map_i, map_j + 1] = cl_deproj_tempeb
                        cl_deproj[map_i + 1, map_j] = cl_deproj_tempbe
                        cl_deproj[map_i + 1, map_j + 1] = cl_deproj_tempb
                        map_j += 2

                if trc[tr_i].spin == 2:
                    map_i += 2
                else:
                    map_i += 1

        return cl_deproj, cl_deproj_bias

    def get_cl_guess(self,ld,cld) :
        """
        Read or compute the guess power spectra.
        :param ld: list of multipoles at which the data power spectra have been measured.
        :param cld: list of power spectrum measurements from the data.
        """

        if self.config['guess_spectrum']=='NONE' :
            print("Interpolating data power spectra")
            l_use=ld
            cl_use=cld
        else:
            data=np.loadtxt(self.config['guess_spectrum'],unpack=True)
            l_use=data[0]
            cl_use=data[1:]
            if cl_use.shape != (self.nmaps, self.nmaps, self.nell):
                raise ValueError("Theory power spectra have a wrong shape.")
        #Interpolate
        lth=np.arange(2,self.lmax+1)

        clth = np.zeros((self.nmaps, self.nmaps, lth.shape[0]))
        for i in range(self.nmaps):
            for ii in range(i, self.nmaps):
                clf = interp1d(l_use, cl_use[i, ii], bounds_error=False, fill_value=0, kind='linear')
                clth[i, ii, :] = clf(lth)
                clth[i, ii, lth <= l_use[0]] = cl_use[i, ii, 0]
                clth[i, ii, lth >= l_use[-1]] = cl_use[i, ii, -1]

        return lth, clth

    def get_power_spectra(self,trc,wsp,bpws) :
        """
        Compute all possible power spectra between pairs of tracers
        :param trc: list of Tracers.
        :param wsp: NaMaster workspace.
        :param bpws: NaMaster bandpowers.
        """

        cls_decoupled = np.zeros((self.nmaps, self.nmaps, self.nell))
        cls_coupled = np.zeros_like(cls_decoupled)

        map_i = 0
        map_j = 0
        for tr_i in range(self.ntracers) :
            for tr_j in range(tr_i, self.ntracers) :
                cl_coupled_temp = nmt.compute_coupled_cell_flat(trc[tr_i].field,trc[tr_j].field,bpws)
                cl_decoupled_temp = wsp[tr_i, tr_j].decouple_cell(cl_coupled_temp)
                if trc[tr_i].spin == 0 and trc[tr_j].spin == 0:
                    cls_coupled[map_i, map_j] = cl_coupled_temp[0]
                    cls_decoupled[map_i, map_j] = cl_decoupled_temp[0]
                    map_j += 1
                elif trc[tr_i].spin == 0 and trc[tr_j].spin == 2:
                    # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                    cl_coupled_tempe = cl_coupled_temp[0]
                    cl_coupled_tempb = cl_coupled_temp[1]
                    cl_decoupled_tempe = cl_decoupled_temp[0]
                    cl_decoupled_tempb = cl_decoupled_temp[1]
                    cls_coupled[map_i, map_j] = cl_coupled_tempe
                    cls_coupled[map_i, map_j+1] = cl_coupled_tempb
                    cls_decoupled[map_i, map_j] = cl_decoupled_tempe
                    cls_decoupled[map_i, map_j+1] = cl_decoupled_tempb
                    map_j += 2
                elif trc[tr_i].spin == 2 and trc[tr_j].spin == 0:
                    # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                    cl_coupled_tempe = cl_coupled_temp[0]
                    cl_coupled_tempb = cl_coupled_temp[1]
                    cl_decoupled_tempe = cl_decoupled_temp[0]
                    cl_decoupled_tempb = cl_decoupled_temp[1]
                    cls_coupled[map_i, map_j] = cl_coupled_tempe
                    cls_coupled[map_i+1, map_j] = cl_coupled_tempb
                    cls_decoupled[map_i, map_j] = cl_decoupled_tempe
                    cls_decoupled[map_i+1, map_j] = cl_decoupled_tempb
                    map_j += 1
                else:
                    # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
                    cl_coupled_tempe = cl_coupled_temp[0]
                    cl_coupled_tempeb = cl_coupled_temp[1]
                    cl_coupled_tempbe = cl_coupled_temp[2]
                    cl_coupled_tempb = cl_coupled_temp[3]
                    cl_decoupled_tempe = cl_decoupled_temp[0]
                    cl_decoupled_tempeb = cl_decoupled_temp[1]
                    cl_decoupled_tempbe = cl_decoupled_temp[2]
                    cl_decoupled_tempb = cl_decoupled_temp[3]
                    cls_coupled[map_i, map_j] = cl_coupled_tempe
                    cls_coupled[map_i+1, map_j] = cl_coupled_tempeb
                    cls_coupled[map_i, map_j+1] = cl_coupled_tempbe
                    cls_coupled[map_i+1, map_j+1] = cl_coupled_tempb
                    cls_decoupled[map_i, map_j] = cl_decoupled_tempe
                    cls_decoupled[map_i+1, map_j] = cl_decoupled_tempeb
                    cls_decoupled[map_i, map_j+1] = cl_decoupled_tempbe
                    cls_decoupled[map_i+1, map_j+1] = cl_decoupled_tempb
                    map_j += 2

            if trc[tr_i].spin == 2:
                map_i += 2
            else:
                map_i += 1

        return cls_decoupled, cls_coupled

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

        logger.info("Computing MCM.")
        wsps = [[0 for i in range(self.ntracers)] for ii in range(self.ntracers)]

        if self.get_input('ngal_maps') != 'NONE':
            logger.info('Number density maps provided.')
            # Compute wsp for counts (is always the same as mask is the same)
            wsp_counts = nmt.NmtWorkspaceFlat()
            if not os.path.isfile(self.get_output_fname('mcm',ext='dat')[0]) :
                logger.info("Computing MCM for counts.")
                wsp_counts.compute_coupling_matrix(tracers[0].field,tracers[0].field,bpws)
                wsp_counts.write_to(self.get_output_fname('mcm',ext='dat')[0])
            else:
                logger.info("Reading MCM for counts.")
                wsp_counts.read_from(self.get_output_fname('mcm',ext='dat')[0])

            if self.get_input('shear_maps') != 'NONE':
                logger.info('Number density and shear maps provided.')
                for i in range(self.ntracers):
                    for ii in range(i, self.ntracers):
                        if i < self.ntracers_counts and ii < self.ntracers_counts:
                            wsps[i, ii] = wsp_counts
                        elif i == 0 and ii >= self.ncounts_maps:
                            wsp = nmt.NmtWorkspaceFlat()
                            if not os.path.isfile(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]]):
                                logger.info("Computing MCM for counts x shear.")
                                wsp.compute_coupling_matrix(tracers[i].field, tracers[ii].field, bpws)
                                wsp.write_to(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]])
                            else:
                                logger.info("Reading MCM for counts x shear.")
                                wsp.read_from(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]])
                            wsps[i, ii] = wsp
                        elif i != 0 and i < self.ntracers_counts and ii >= self.ntracers_counts:
                            wsps[i, ii] = wsps[0, ii]
                        elif i >= self.ntracers_counts and ii >= self.ntracers_counts:
                            wsp = nmt.NmtWorkspaceFlat()
                            if not os.path.isfile(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]]):
                                logger.info("Computing MCM for shear.")
                                wsp.compute_coupling_matrix(tracers[i].field, tracers[ii].field, bpws)
                                wsp.write_to(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]])
                            else:
                                logger.info("Reading MCM for shear.")
                                wsp.read_from(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]])
                            wsps[i, ii] = wsp
                        else:
                            raise RuntimeError("Messed-up indexing in wsp computation.")
        # Only shear maps case
        else:
            logger.info('Shear maps provided.')
            for i in range(self.ntracers):
                for ii in range(i, self.ntracers):
                    wsp = nmt.NmtWorkspaceFlat()
                    if not os.path.isfile(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]]):
                        logger.info("Computing MCM for shear.")
                        wsp.compute_coupling_matrix(tracers[i].field, tracers[ii].field, bpws)
                        wsp.write_to(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]])
                    else:
                        logger.info("Reading MCM for shear.")
                        wsp.read_from(self.get_output_fname('mcm', ext='dat')[self.ordering[i, ii]])
                    wsps[i, ii] = wsp

        return wsps

    def get_covar_mcm(self,tracers,bpws):
        """
        Get NmtCovarianceWorkspaceFlat for our mask
        """
        cwsp=nmt.NmtCovarianceWorkspaceFlat()
        if not os.path.isfile(self.get_output_fname('cov_mcm',ext='dat')) :
            print("Computing covariance MCM")
            cwsp = [[[[0 for i in range(self.nmaps)] for ii in self.nmaps]
                     for j in range(self.nmaps)] for jj in range(self.nmaps)]
            # Compute wsp for counts (is always the same as mask is the same)
            cwsp_counts = nmt.NmtCovarianceWorkspaceFlat()
            print("Computing covariance MCM for counts.")
            cwsp_counts.compute_coupling_coefficients(tracers[0].field, tracers[0].field, bpws)
            cwsp_counts.write_to(self.get_output_fname('cov_mcm', ext='dat')[0])

            for i1 in range(self.nmaps):
                for j1 in range(i1, self.nmaps):
                    for i2 in range(self.nmaps):
                        for j2 in range(i2, self.nmaps):
                            tr_i1, tr_j1 = self.pss2tracers[i1, j1]
                            tr_i2, tr_j2 = self.pss2tracers[i2, j2]
                            tr_indxs = np.array([tr_i1, tr_j1, tr_i2, tr_j2])
                            if np.all(tr_indxs < self.ntracers_counts):
                                cwsp_curr = cwsp_counts
                            elif np.any(tr_indxs < self.ntracers_counts) and np.all(tr_indxs != 0):
                                i1_curr = i1
                                j1_curr = j1
                                i2_curr = i2
                                j2_curr = j2
                                if tr_i1 < self.ntracers_counts:
                                    i1_curr = 0
                                if tr_j1 < self.ntracers_counts:
                                    j1_curr = 0
                                if tr_i2 < self.ntracers_counts:
                                    i2_curr = 0
                                if tr_j2 < self.ntracers_counts:
                                    j2_curr = 0
                                cwsp_curr = cwsp[i1_curr, j1_curr, i2_curr, j2_curr]
                            else:
                                cwsp_curr = nmt.NmtCovarianceWorkspaceFlat()
                                cwsp_curr.compute_coupling_coefficients(tracers[tr_i1].field, tracers[tr_j1].field, bpws,
                                                                        tracers[tr_i2].field, tracers[tr_j2].field, bpws)

                            cwsp[i1, j1, i2, j2] = cwsp_curr
                            cwsp_curr.write_to(self.get_output_fname('cov_mcm', ext='dat')[0])
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

        covar=np.zeros([self.ncross, self.nell, self.ncross, self.nell])
        # Get covar MCM for counts tracers
        cwsp=self.get_covar_mcm(tracers,bpws)

        ix_1=0
        for i1 in enumerate(self.nmaps) :
            for j1 in enumerate(i1, self.nmaps) :
                ix_2=0
                for i2 in enumerate(self.nmaps) :
                    for j2 in enumerate(i2, self.nmaps) :
                        tr_i1, tr_j1 = self.pss2tracers[i1, j1]
                        tr_i2, tr_j2 = self.pss2tracers[i2, j2]

                        ca1b1=clth[i1, i2]
                        ca1b2=clth[i1, j2]
                        ca2b1=clth[j1, i2]
                        ca2b2=clth[j1, j2]
                        cov_here=nmt.gaussian_covariance_flat(cwsp[i1, j1, i2, j2], tracers[tr_i1].spin, tracers[tr_i2].spin,
                                                              tracers[tr_j1].spin, tracers[tr_j2].spin, lth,
                                                              [ca1b1], [ca1b2], [ca2b1], [ca2b2], wsp[tr_i1, tr_j1],
                                                              wsp[tr_i2, tr_j2])
                        covar[ix_1, :, ix_2, :] = cov_here
                        ix_2+=1
                ix_1+=1

        covar = covar.reshape([self.ncross*self.nell, self.ncross*self.nell])

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

    def get_tracers(self, temps, map_type='ngal_maps') :
        """
        Produce a Tracer for each redshift bin. Do so with and without contaminant deprojection.
        :param temps: list of contaminant tracers
        """
        hdul=fits.open(self.get_input(map_type))

        if map_type == 'ngal_maps':
            logger.info('Creating number counts tracers.')
            if len(hdul)%2!=0 :
                raise ValueError("Input file should have two HDUs per map")
            nmaps=len(hdul)//2
            tracers_nocont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None)
                            for i in range(nmaps)]
            tracers_wcont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=temps)
                           for i in range(nmaps)]

        elif map_type == 'shear_maps':
            logger.info('Creating shear tracers.')
            if len(hdul)%6!=0 :
                raise ValueError("Input file should have six HDUs per map")
            nmaps=len(hdul)//6
            tracers_nocont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, is_shear=True, weightmask=True)
                            for i in range(nmaps)]
            tracers_wcont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, is_shear=True, weightmask=True)
                           for i in range(nmaps)]

        else:
            raise NotImplementedError()

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

        for i_t,t in enumerate(tracers):
            if i_t < self.ntracers_counts:
                # z = (t.nz_data['z_i'] + t.nz_data['z_f']) * 0.5
                # nz = t.nz_data['nz_cosmos']
                # tracer = sacc.tracers.BaseTracer.make('NZ',
                #                                       'gc_{}'.format(i_t),
                #                                       'delta_g',
                #                                       spin=0,
                #                                       z=z,
                #                                       nz=nz,
                #                                       extra_columns={'nz_'+c: t.nz_data['nz_'+c]
                #                                         for c in ['demp','ephor','ephor_ab','frankenz','nnpz']})
                tracer = sacc.tracers.BaseTracer.make('NZ',
                                                      'gc_{}'.format(i_t),
                                                      'delta_g',
                                                      spin=0)
            else:
                # z = (t.nz_data['z_i'] + t.nz_data['z_f']) * 0.5
                # nz = t.nz_data['nz_cosmos']
                # tracer = sacc.tracers.BaseTracer.make('NZ',
                #                                       'wl_{}'.format(i_t-self.ntracers_counts),
                #                                       'cosmic_shear',
                #                                       spin=2,
                #                                       z=z,
                #                                       nz=nz,
                #                                       extra_columns={'nz_'+c: t.nz_data['nz_'+c]
                #                                         for c in ['demp','ephor','ephor_ab','frankenz','nnpz']})
                tracer = sacc.tracers.BaseTracer.make('NZ',
                                                      'wl_{}'.format(i_t-self.ntracers_counts),
                                                      'cosmic_shear',
                                                      spin=2,
                                                      z=np.linspace(0, 1, 100),
                                                      nz=np.ones(100))

            sacc_tracers.append(tracer)

        return sacc_tracers

    def write_vector_to_sacc(self, fname_out, sacc_t, cls, windows, covar=None) :
        """
        Write a vector of power spectrum measurements into a SACC file.
        :param fname_out: path to output file
        :param sacc_t: list of SACC tracers
        :param sacc_b: SACC Binning object.
        :param cls: list of power spectrum measurements.
        :param covar: covariance matrix:
        :param verbose: do you want to print out information about the SACC file?
        """

        # Add tracers to sacc
        saccfile = sacc.Sacc()
        for trc in sacc_t:
            saccfile.add_tracer_object(trc)

        map_i = 0
        map_j = 0
        for tr_i in range(self.ntracers):
            for tr_j in range(tr_i, self.ntracers):
                if sacc_t[tr_i].spin == 0 and sacc_t[tr_j].spin == 0:
                    saccfile.add_ell_cl('cl_00',
                                 'gc_{}'.format(tr_i),
                                 'gc_{}'.format(tr_j),
                                 ells,
                                 cls[map_i, map_j, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell)
                                 )
                    map_j += 1
                elif sacc_t[tr_i].spin == 0 and sacc_t[tr_j].spin == 2:
                    saccfile.add_ell_cl('cl_0e',
                                 'gc_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j),
                                 ells,
                                 cls[map_i, map_j, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell))
                    saccfile.add_ell_cl('cl_0b',
                                 'gc_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j),
                                 ells,
                                 cls[map_i, map_j+1, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell))
                    map_j += 2
                elif sacc_t[tr_i].spin == 2 and sacc_t[tr_j].spin == 0:
                    saccfile.add_ell_cl('cl_0e',
                                        'wl_{}'.format(tr_i),
                                        'gc_{}'.format(tr_j),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=windows[tr_i, tr_j],
                                        window_id=range(n_ell))
                    saccfile.add_ell_cl('cl_0b',
                                        'wl_{}'.format(tr_i),
                                        'gc_{}'.format(tr_j),
                                        ells,
                                        cls[map_i+1, map_j, :],
                                        window=windows[tr_i, tr_j],
                                        window_id=range(n_ell))
                    map_j += 1
                else:
                    saccfile.add_ell_cl('cl_ee',
                                 'wl_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j),
                                 ells,
                                 cls[map_i, map_j, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell))
                    saccfile.add_ell_cl('cl_eb',
                                 'wl_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j),
                                 ells,
                                 cls[map_i+1, map_j, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell))
                    saccfile.add_ell_cl('cl_be',
                                 'wl_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j),
                                 ells,
                                 cls[map_i, map_j+1, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell))
                    saccfile.add_ell_cl('cl_bb',
                                 'wl_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j),
                                 ells,
                                 cls[map_i+1, map_j+1, :],
                                 window=windows[tr_i, tr_j],
                                 window_id=range(n_ell))
                    map_j += 2

                if sacc_t[tr_i].spin == 2:
                    map_i += 2
                else:
                    map_i += 1

        if covar is not None :
            saccfile.add_covariance(covar)

        saccfile.save_fits(fname_out, overwrite=True)

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

    def mapping(self, trcs):

        self.pss2tracers = [[0 for i in range(self.nmaps)] for ii in range(self.nmaps)]
        self.maps2tracers = [0 for i in range(self.nmaps)]

        map_i = 0
        map_j = 0
        for tr_i in range(self.ntracers) :
            self.maps2tracers[map_i] = tr_i
            for tr_j in range(tr_i, self.ntracers) :
                if trcs[tr_i].spin == 0 and trcs[tr_j].spin == 0:
                    self.pss2tracers[map_i, map_j] = (tr_i, tr_j)
                    map_j += 1
                elif trcs[tr_i].spin == 0 and trcs[tr_j].spin == 2:
                    # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                    self.pss2tracers[map_i, map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i, map_j+1] = (tr_i, tr_j)
                    map_j += 2
                elif trcs[tr_i].spin == 2 and trcs[tr_j].spin == 0:
                    # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                    self.pss2tracers[map_i, map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i+1, map_j] = (tr_i, tr_j)
                    map_j += 1
                else:
                    # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
                    self.pss2tracers[map_i, map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i+1, map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i, map_j+1] = (tr_i, tr_j)
                    self.pss2tracers[map_i+1, map_j+1] = (tr_i, tr_j)
                    map_j += 2

            if trcs[tr_i].spin == 2:
                map_i += 2
            else:
                map_i += 1
                
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

        logger.info("Reading mask.")
        self.msk_bi,self.mskfrac,self.mp_depth=self.get_masks()

        logger.info("Computing area.")
        self.area_pix=np.radians(self.fsk.dx)*np.radians(self.fsk.dy)
        self.area_patch=np.sum(self.msk_bi*self.mskfrac)*self.area_pix
        self.lmax=int(180.*np.sqrt(1./self.fsk.dx**2+1./self.fsk.dy**2))

        logger.info("Reading contaminants.")
        temps=self.get_contaminants()

        logger.info("Setting bandpowers.")
        lini=np.array(self.config['ell_bpws'])[:-1]
        lend=np.array(self.config['ell_bpws'])[ 1:]
        bpws=nmt.NmtBinFlat(lini,lend)
        ell_eff=bpws.get_effective_ells()

        if self.get_input('ngal_maps') != 'NONE' or self.get_input('shear_maps') != 'NONE':
            if self.get_input('ngal_maps') != 'NONE':
                logger.info("Generating number density tracers.")
                tracers_nc,tracers_wc=self.get_tracers(temps, map_type='ngal_maps')
                self.ntracers_counts = len(tracers_nc)
            else:
                logger.info("No number density maps provided.")
                self.ntracers_counts = 0
                tracers_nc = []
                tracers_wc = []
            if self.get_input('shear_maps') != 'NONE':
                logger.info("Generating shear tracers.")
                tracers_shear_nc, tracers_shear_wc = self.get_tracers(temps, map_type='shear_maps')
                self.ntracers_shear = len(tracers_shear_nc)

                logger.info("Appending shear tracers to number density tracers.")
                tracers_nc.extend(tracers_shear_nc)
                tracers_wc.extend(tracers_shear_wc)
            else:
                self.ntracers_shear = 0
        else:
            raise RuntimeError('Either ngal_maps or shear_maps need to be provided. Aborting.')

        self.ntracers = len(tracers_nc)
        self.nmaps = self.ntracers_counts + 2*self.ntracers_shear

        logger.info("Translating into SACC tracers.")
        tracers_sacc=self.get_sacc_tracers(tracers_nc)

        self.ordering = np.zeros([self.nmaps,self.nmaps],dtype=int)
        ix=0
        for i in range(self.nmaps) :
            for j in range(i,self.nmaps) :
                self.ordering[i,j]=ix
                if j!=i :
                    self.ordering[j,i]=ix
                ix+=1

        logger.info("Getting MCM.")
        wsp = self.get_mcm(tracers_nc,bpws)

        logger.info("Computing window function.")
        windows = self.get_windows(wsp)

        logger.info("Computing power spectra.")
        logger.info(" No deprojections.")
        cls_wodpj,_=self.get_power_spectra(tracers_nc,wsp,bpws)
        logger.info(" W. deprojections.")
        cls_wdpj,cls_wdpj_coupled=self.get_power_spectra(tracers_wc,wsp,bpws)
        self.ncross,self.nell=cls_wodpj.shape

        logger.info("Getting guess power spectra.")
        lth,clth=self.get_cl_guess(ell_eff,cls_wdpj)

        logger.info("Computing deprojection bias.")
        cls_wdpj,cls_deproj=self.get_dpj_bias(tracers_wc,lth,clth,cls_wdpj_coupled,wsp,bpws)

        logger.info("Computing covariance.")
        cov_wodpj=self.get_covar(lth,clth,bpws,tracers_wc,wsp,None,None)
        if self.config['gaus_covar_type']=='analytic' :
            cov_wdpj=cov_wodpj.copy()
        else :
            cov_wdpj=self.get_covar(lth,clth,bpws,tracers_wc,wsp,temps,cls_deproj)

        logger.info("Computing noise bias.")
        nls=self.get_noise(tracers_nc,wsp,bpws)

        logger.info("Writing output")
        self.write_vector_to_sacc(self.get_output_fname('noi_bias',ext='sacc'),tracers_sacc,
                                  nls,windows)
        self.write_vector_to_sacc(self.get_output_fname('dpj_bias',ext='sacc'),tracers_sacc,
                                  cls_deproj,windows)
        self.write_vector_to_sacc(self.get_output_fname('power_spectra_wodpj',ext='sacc'),tracers_sacc,
                                  cls_wodpj,windows,covar=cov_wodpj)
        self.write_vector_to_sacc(self.get_output_fname('power_spectra_wdpj',ext='sacc'),tracers_sacc,
                                  cls_wdpj,windows,covar=cov_wdpj)

if __name__ == '__main__':
    cls = PipelineStage.main()
