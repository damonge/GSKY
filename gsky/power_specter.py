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
            ('act_maps', FitsFile), ('dust_map',FitsFile),('star_map',FitsFile),
            ('depth_map',FitsFile),('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),
            ('exptime_maps',FitsFile),('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),
            ('seeing_maps',FitsFile),('ellipt_maps',FitsFile),('nvisit_maps',FitsFile),
            ('cosmos_weights',FitsFile),('syst_masking_file',ASCIIFile)]
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
                    'output_run_dir': 'NONE','sys_collapse_type':'average'}

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

    def get_windows(self, tracers, wsp):
        """
        Get window functions for each bandpower so they can be stored into the final SACC files.
        """

        # Compute window functions
        logger.info("Computing window functions.")
        l_arr = np.arange(self.lmax + 1)

        windows_list = [[0 for i in range(self.ntracers)] for ii in range(self.ntracers)]

        zero_arr = np.zeros(self.lmax + 1)

        tracer_type_arr = [tr.type for tr in tracers]

        for i in range(self.ntracers):
            for ii in range(i, self.ntracers):

                # File does not exist
                if not os.path.isfile(self.get_output_fname('windows_l') + '_{}{}'.format(i, ii) + '.npz'):
                    tr_types_cur = [tracers[i].type, tracers[ii].type]
                    # All galaxy maps
                    if set(tr_types_cur) == {'delta_g', 'delta_g'}:
                        if not hasattr(self, 'windows_counts'):
                            counts_indx = tracer_type_arr.index('delta_g')
                            if not os.path.isfile(self.get_output_fname('windows_l')+'_{}{}'.format(counts_indx, counts_indx)+'.npz'):
                                logger.info("Computing window functions for counts.")
                                self.windows_counts = np.zeros([nbands, self.lmax + 1])
                                t_hat = np.zeros(self.lmax + 1)
                                for il, l in enumerate(l_arr):
                                    t_hat[il] = 1.
                                    self.windows_counts[:, il] = wsp[counts_indx][counts_indx].decouple_cell(wsp[counts_indx][counts_indx].couple_cell(l_arr, [t_hat]))
                                    t_hat[il] = 0.
                                np.savez(self.get_output_fname('windows_l')+'_{}{}'.format(counts_indx, counts_indx)+'.npz', windows=self.windows_counts)
                                logger.info('Written window function to {}.'.format(
                                    self.get_output_fname('windows_l') + '_{}{}'.format(counts_indx, counts_indx) + '.npz'))
                            else:
                                logger.info("Reading window functions for counts.")
                                self.windows_counts = np.load(self.get_output_fname('windows_l')+'_{}{}'.format(counts_indx, counts_indx)+'.npz')['windows']
                                logger.info('Read window function from {}.'.format(
                                    self.get_output_fname('windows_l') + '_{}{}'.format(counts_indx,
                                                                                        counts_indx) + '.npz'))
                        windows_curr = self.windows_counts

                    # One galaxy map
                    elif 'delta_g' in tr_types_cur:
                        counts_indx = tracer_type_arr.index('delta_g')
                        i_curr = i
                        ii_curr = ii
                        if tracers[i].type == 'delta_g':
                            i_curr = counts_indx
                        if tracers[ii].type == 'delta_g':
                            ii_curr = counts_indx
                        if not os.path.isfile(self.get_output_fname('windows_l')+'_{}{}'.format(i_curr, ii_curr)+'.npz'):
                            logger.info("Computing window functions for counts xcorr.")
                            windows_curr = np.zeros([nbands, self.lmax + 1])
                            t_hat = np.zeros(self.lmax + 1)
                            if 'cosmic_shear' in tr_types_cur:
                                logger.info("Only using E-mode window function.")
                                for il, l in enumerate(l_arr):
                                    t_hat[il] = 1.
                                    windows_curr[:, il] = wsp[i_curr][ii_curr].decouple_cell(wsp[i_curr][ii_curr].couple_cell(l_arr, [t_hat, zero_arr]))[0, :]
                                    t_hat[il] = 0.
                            else:
                                for il, l in enumerate(l_arr):
                                    t_hat[il] = 1.
                                    windows_curr[:, il] = wsp[i_curr][ii_curr].decouple_cell(wsp[i_curr][ii_curr].couple_cell(l_arr, [t_hat]))
                                    t_hat[il] = 0.
                            np.savez(self.get_output_fname('windows_l')+'_{}{}'.format(i_curr, ii_curr)+'.npz', windows=windows_curr)
                            logger.info('Written window function to {}.'.format(self.get_output_fname('windows_l')+'_{}{}'.format(i_curr, ii_curr)+'.npz'))
                        else:
                            logger.info("Reading window functions for counts xcorr.")
                            windows_curr = np.load(self.get_output_fname('windows_l')+ '_{}{}'.format(i_curr, ii_curr) + '.npz')['windows']
                            logger.info('Read window function from {}.'.format(
                                self.get_output_fname('windows_l') + '_{}{}'.format(i_curr, ii_curr) + '.npz'))

                    # No galaxy maps
                    else:
                        logger.info("Computing window functions for {}.".format(self.get_output_fname('windows_l')+'_{}{}'.format(i, ii)+'.npz'))
                        windows_curr = np.zeros([nbands, self.lmax + 1])
                        t_hat = np.zeros(self.lmax + 1)
                        if set(tr_types_cur) == {'cosmic_shear', 'cosmic_shear'}:
                            logger.info("Only using E-mode window function.")
                            for il, l in enumerate(l_arr):
                                t_hat[il] = 1.
                                windows_curr[:, il] = wsp[i][ii].decouple_cell(wsp[i][ii].couple_cell(l_arr, [t_hat, zero_arr, zero_arr, zero_arr]))[0, :]
                                t_hat[il] = 0.
                        elif 'cosmic_shear' in tr_types_cur:
                            logger.info("Only using E-mode window function.")
                            for il, l in enumerate(l_arr):
                                t_hat[il] = 1.
                                windows_curr[:, il] = wsp[i][ii].decouple_cell(wsp[i][ii].couple_cell(l_arr, [t_hat, zero_arr]))[0, :]
                                t_hat[il] = 0.
                        else:
                            for il, l in enumerate(l_arr):
                                t_hat[il] = 1.
                                windows_curr[:, il] = wsp[i][ii].decouple_cell(wsp[i][ii].couple_cell(l_arr, [t_hat]))
                                t_hat[il] = 0.
                        np.savez(self.get_output_fname('windows_l')+ '_{}{}'.format(i, ii) + '.npz', windows=windows_curr)

                # File exists
                else:
                    logger.info("Reading window functions for {}.".format(self.get_output_fname('windows_l')+'_{}{}'.format(i, ii)+'.npz'))
                    windows_curr = np.load(self.get_output_fname('windows_l')+ '_{}{}'.format(i, ii) + '.npz')['windows']

                windows_list[i][ii] = windows_curr

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

        nls = np.zeros((self.nmaps, self.nmaps, self.nbands))

        zero_arr = np.zeros(self.nbands)

        map_i = 0
        for tr_i in range(self.ntracers):
            map_j = map_i
            for tr_j in range(tr_i, self.ntracers):
                if tr_i == tr_j:
                    t = tracers[tr_j]
                    type_cur = t.type
                    logger.info('Computing analytic noise for tracer_type = {}.'.format(type_cur))
                    if type_cur == 'delta_g':

                        corrfac = np.sum(t.weight) / (t.fsk.nx * t.fsk.ny)
                        nl = np.ones(self.nbands) * corrfac / t.ndens_perad

                        nls[map_i, map_j] = wsp[tr_i][tr_j].decouple_cell([nl])[0]
                        map_j += 1
                    elif type_cur == 'cosmic_shear':
                        # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]

                        corrfac = np.sum(t.weight)/(t.fsk.nx*t.fsk.ny)
                        nl = np.ones(self.nbands)*np.mean(t.e1_2rms_cat+t.e2_2rms_cat)*corrfac/t.ndens_perad
                        nls_temp = wsp[tr_i][tr_j].decouple_cell([nl, zero_arr, zero_arr, nl])

                        nls_tempe = nls_temp[0]
                        nls_tempb = nls_temp[3]
                        nls[map_i, map_j] = nls_tempe
                        nls[map_i+1, map_j+1] = nls_tempb
                        map_j += 2
                    elif type_cur == 'Compton_y' or type_cur == 'kappa':
                        logger.info('Setting analytic noise to zero.')
                        nls[map_i, map_j] = np.zeros(self.nbands)

            if type_cur == 'cosmic_shear' == 2:
                map_i += 2
            else:
                map_i += 1

        return nls
        
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
            s = sacc.Sacc.load_fits(self.get_output_fname('dpj_bias',ext='sacc'))
            cl_deproj_bias_mean = s.mean
            cl_deproj_bias = self.convert_sacc_to_clarr(cl_deproj_bias_mean, trc)
            cl_deproj = np.zeros_like(cl_deproj_bias)

            # Remove deprojection bias
            map_i = 0
            for tr_i in range(self.ntracers):
                map_j = map_i
                for tr_j in range(tr_i, self.ntracers):
                    if trc[tr_i].spin == 0 and trc[tr_j].spin == 0:
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j]], cl_bias=[cl_deproj_bias[map_i, map_j]])
                        cl_deproj[map_i, map_j] = cl_deproj_temp[0]
                        map_j += 1
                    elif trc[tr_i].spin == 0 and trc[tr_j].spin == 2:
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i, map_j + 1]],
                                                           cl_bias=[cl_deproj_bias[map_i, map_j], cl_deproj_bias[map_i, map_j+1]])
                        # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                        cl_deproj_tempe = cl_deproj_temp[0]
                        cl_deproj_tempb = cl_deproj_temp[1]
                        cl_deproj[map_i, map_j] = cl_deproj_tempe
                        cl_deproj[map_i, map_j + 1] = cl_deproj_tempb
                        map_j += 2
                    elif trc[tr_i].spin == 2 and trc[tr_j].spin == 0:
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i + 1, map_j]],
                                                           cl_bias=[cl_deproj_bias[map_i, map_j], cl_deproj_bias[map_i+1, map_j]])
                        # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                        cl_deproj_tempe = cl_deproj_temp[0]
                        cl_deproj_tempb = cl_deproj_temp[1]
                        cl_deproj[map_i, map_j] = cl_deproj_tempe
                        cl_deproj[map_i + 1, map_j] = cl_deproj_tempb
                        map_j += 1
                    else:
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i, map_j + 1],
                                                            cl_coupled[map_i + 1, map_j], cl_coupled[map_i + 1, map_j + 1]],
                                                           cl_bias=[cl_deproj_bias[map_i, map_j], cl_deproj_bias[map_i, map_j+1],
                                                                    cl_deproj_bias[map_i+1, map_j], cl_deproj_bias[map_i+1, map_j+1]])
                        # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
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
        else :
            logger.info("Computing deprojection bias.")

            cl_deproj_bias = np.zeros((self.nmaps, self.nmaps, self.nbands))
            cl_deproj = np.zeros_like(cl_deproj_bias)

            # Compute and remove deprojection bias
            map_i = 0
            for tr_i in range(self.ntracers):
                map_j = map_i
                for tr_j in range(tr_i, self.ntracers):
                    if trc[tr_i].spin == 0 and trc[tr_j].spin == 0:
                        cl_deproj_bias_temp = nmt.deprojection_bias_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                                            lth, [clth[map_i, map_j]])
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j]], cl_bias=cl_deproj_bias_temp)
                        cl_deproj_bias[map_i, map_j] = cl_deproj_bias_temp[0]
                        cl_deproj[map_i, map_j] = cl_deproj_temp[0]
                        map_j += 1
                    elif trc[tr_i].spin == 0 and trc[tr_j].spin == 2:
                        cl_deproj_bias_temp = nmt.deprojection_bias_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                                lth, [clth[map_i, map_j], clth[map_i, map_j + 1]])
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i, map_j + 1]],
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
                    elif trc[tr_i].spin == 2 and trc[tr_j].spin == 0:
                        cl_deproj_bias_temp = nmt.deprojection_bias_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                                lth, [clth[map_i, map_j], clth[map_i + 1, map_j]])
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i + 1, map_j]],
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
                        cl_deproj_bias_temp = nmt.deprojection_bias_flat(trc[tr_i].field, trc[tr_j].field, bpws,
                                                        lth, [clth[map_i, map_j], clth[map_i, map_j + 1],
                                                              clth[map_i + 1, map_j], clth[map_i + 1, map_j + 1]])
                        cl_deproj_temp = wsp[tr_i][tr_j].decouple_cell([cl_coupled[map_i, map_j], cl_coupled[map_i, map_j + 1],
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
            if cl_use.shape != (self.nmaps, self.nmaps, self.nbands):
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

                if i != ii:
                    clth[ii, i, :] = clth[i, ii, :]

        return lth, clth

    def get_power_spectra(self,trc,wsp,bpws) :
        """
        Compute all possible power spectra between pairs of tracers
        :param trc: list of Tracers.
        :param wsp: NaMaster workspace.
        :param bpws: NaMaster bandpowers.
        """

        cls_decoupled = np.zeros((self.nmaps, self.nmaps, self.nbands))
        cls_coupled = np.zeros_like(cls_decoupled)

        map_i = 0
        for tr_i in range(self.ntracers) :
            map_j = map_i
            for tr_j in range(tr_i, self.ntracers) :
                cl_coupled_temp = nmt.compute_coupled_cell_flat(trc[tr_i].field,trc[tr_j].field,bpws)
                cl_decoupled_temp = wsp[tr_i][tr_j].decouple_cell(cl_coupled_temp)
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
            
    def get_mcm(self,tracers,bpws) :
        """
        Get NmtWorkspaceFlat for our mask
        """

        logger.info("Computing MCM.")
        wsps = [[0 for i in range(self.ntracers)] for ii in range(self.ntracers)]

        tracer_type_arr = [tr.type for tr in tracers]

        for i in range(self.ntracers):
            for ii in range(i, self.ntracers):

                # File does not exist
                if not os.path.isfile(self.get_output_fname('mcm') + '_{}{}'.format(i, ii) + '.dat'):
                    tr_types_cur = [tracers[i].type, tracers[ii].type]
                    # All galaxy maps
                    if set(tr_types_cur) == {'delta_g', 'delta_g'}:
                        if not hasattr(self, 'wsp_counts'):
                            counts_indx = tracer_type_arr.index('delta_g')
                            wsp_curr = nmt.NmtWorkspaceFlat()
                            if not os.path.isfile(self.get_output_fname('mcm') + '_{}{}'.format(counts_indx, counts_indx) + '.dat'):
                                logger.info("Computing MCM for counts.")
                                wsp_curr.compute_coupling_matrix(tracers[counts_indx].field, tracers[counts_indx].field, bpws)
                                wsp_curr.write_to(self.get_output_fname('mcm') + '_{}{}'.format(counts_indx, counts_indx) + '.dat')
                                logger.info("MCM written to {}.".format(self.get_output_fname('mcm') + '_{}{}'.format(counts_indx, counts_indx) + '.dat'))
                            else:
                                logger.info("Reading MCM for counts.")
                                wsp_curr.read_from(self.get_output_fname('mcm') + '_{}{}'.format(counts_indx, counts_indx) + '.dat')
                                logger.info("MCM read from {}.".format(
                                    self.get_output_fname('mcm') + '_{}{}'.format(counts_indx, counts_indx) + '.dat'))
                            self.wsp_counts = wsp_curr
                        wsp_curr = self.wsp_counts

                    # One galaxy map
                    elif 'delta_g' in tr_types_cur:
                        counts_indx = tracer_type_arr.index('delta_g')
                        i_curr = i
                        ii_curr = ii
                        if tracers[i].type == 'delta_g':
                            i_curr = counts_indx
                        if tracers[ii].type == 'delta_g':
                            ii_curr = counts_indx
                        wsp_curr = nmt.NmtWorkspaceFlat()
                        if not os.path.isfile(
                                self.get_output_fname('mcm') + '_{}{}'.format(i_curr, ii_curr) + '.dat'):
                            logger.info("Computing MCM for counts xcorr.")
                            wsp_curr.compute_coupling_matrix(tracers[i_curr].field, tracers[ii_curr].field, bpws)
                            wsp_curr.write_to(self.get_output_fname('mcm') + '_{}{}'.format(i_curr, ii_curr) + '.dat')
                            logger.info("MCM written to {}.".format(
                                self.get_output_fname('mcm') + '_{}{}'.format(i_curr, ii_curr) + '.dat'))
                        else:
                            logger.info("Reading MCM for counts xcorr.")
                            wsp_curr.read_from(
                                self.get_output_fname('mcm') + '_{}{}'.format(i_curr, ii_curr) + '.dat')
                            logger.info("MCM read from {}.".format(
                                self.get_output_fname('mcm') + '_{}{}'.format(i_curr, ii_curr) + '.dat'))

                    # No galaxy maps
                    else:
                        logger.info("Computing MCM for {}.".format(self.get_output_fname('mcm') + '_{}{}'.format(i, ii) + '.dat'))
                        wsp_curr = nmt.NmtWorkspaceFlat()
                        wsp_curr.compute_coupling_matrix(tracers[i].field, tracers[ii].field, bpws)
                        wsp_curr.write_to(self.get_output_fname('mcm') + '_{}{}'.format(i, ii) + '.dat')

                # File exists
                else:
                    logger.info("Reading MCM for {}.".format(self.get_output_fname('mcm') + '_{}{}'.format(i, ii) + '.dat'))
                    wsp_curr = nmt.NmtWorkspaceFlat()
                    wsp_curr.read_from(self.get_output_fname('mcm') + '_{}{}'.format(i, ii) + '.dat')

                wsps[i][ii] = wsp_curr

        return wsps

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
        if map_type != 'Compton_y_maps' and map_type != 'kappa_maps':
            hdul=fits.open(self.get_input(map_type))
        else:
            hdul = fits.open(self.get_input('act_maps'))

        if map_type == 'ngal_maps':
            logger.info('Creating number counts tracers.')
            if len(hdul)%2!=0 :
                raise ValueError("Input file should have two HDUs per map")
            nmaps=len(hdul)//2
            tracers_nocont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, type=map_type)
                            for i in range(nmaps)]
            tracers_wcont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=temps, type=map_type)
                           for i in range(nmaps)]

        elif map_type == 'shear_maps':
            logger.info('Creating cosmic shear tracers.')
            if (len(hdul)-1)%6!=0 :
                raise ValueError("Input file should have six HDUs per map")
            nmaps=(len(hdul)-1)//6
            tracers_nocont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, type=map_type, weightmask=True)
                            for i in range(nmaps)]
            tracers_wcont=[Tracer(hdul,i,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, type=map_type, weightmask=True)
                           for i in range(nmaps)]

        elif map_type == 'Compton_y_maps':
            logger.info('Creating Compton_y tracers.')

            tracers_nocont=[Tracer(hdul,0,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, type=map_type)]
            tracers_wcont=[Tracer(hdul,0,self.fsk,self.msk_bi,self.mskfrac,contaminants=temps, type=map_type)]

        elif map_type == 'kappa_maps':
            logger.info('Creating kappa tracers.')

            tracers_nocont=[Tracer(hdul,2,self.fsk,self.msk_bi,self.mskfrac,contaminants=None, type=map_type)]
            tracers_wcont=[Tracer(hdul,2,self.fsk,self.msk_bi,self.mskfrac,contaminants=temps, type=map_type)]

        else:
            raise NotImplementedError()

        hdul.close()

        return tracers_nocont,tracers_wcont

    def get_all_tracers(self, temps):

        if self.get_input('ngal_maps') != 'NONE' or self.get_input('shear_maps') != 'NONE' or self.get_input('Compton_y_maps') != 'NONE':
            if self.get_input('ngal_maps') != 'NONE':
                logger.info("Generating number density tracers.")
                tracers_nc,tracers_wc=self.get_tracers(temps, map_type='ngal_maps')
                self.ntracers_counts = len(tracers_nc)
            else:
                logger.info("No number density maps provided.")
                self.ntracers_counts = 0
                tracers_nc = []
                tracers_wc = []

            if self.get_input('act_maps') != 'NONE':
                logger.info("Generating Compton_y tracers.")
                tracers_comptony_nc, tracers_comptony_wc = self.get_tracers(temps, map_type='Compton_y_maps')
                self.ntracers_comptony = len(tracers_comptony_nc)

                logger.info("Appending Compton_y tracers to number density tracers.")
                tracers_nc.extend(tracers_comptony_nc)
                tracers_wc.extend(tracers_comptony_wc)

                logger.info("Generating kappa tracers.")
                tracers_kappa_nc, tracers_kappa_wc = self.get_tracers(temps, map_type='kappa_maps')
                self.ntracers_kappa = len(tracers_kappa_nc)

                logger.info("Appending kappa tracers to number density tracers.")
                tracers_nc.extend(tracers_kappa_nc)
                tracers_wc.extend(tracers_kappa_wc)

            else:
                logger.info("No Compton_y maps provided.")
                self.ntracers_comptony = 0

                logger.info("No kappa maps provided.")
                self.ntracers_kappa = 0

            if self.get_input('shear_maps') != 'NONE':
                logger.info("Generating shear tracers.")
                tracers_shear_nc, tracers_shear_wc = self.get_tracers(temps, map_type='shear_maps')
                self.ntracers_shear = len(tracers_shear_nc)

                logger.info("Appending shear tracers to number density tracers.")
                tracers_nc.extend(tracers_shear_nc)
                tracers_wc.extend(tracers_shear_wc)
            else:
                logger.info("No shear maps provided.")
                self.ntracers_shear = 0

        else:
            raise RuntimeError('ngal_maps, Compton_y_maps or shear_maps need to be provided. Aborting.')

        return tracers_nc, tracers_wc

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
        if self.config['output_run_dir'] != 'NONE':
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
            if t.type == 'delta_g':
                z = (t.nz_data['z_i'] + t.nz_data['z_f']) * 0.5
                nz = t.nz_data['nz_cosmos']
                tracer = sacc.tracers.BaseTracer.make('NZ',
                                                      'gc_{}'.format(i_t),
                                                      'delta_g',
                                                      spin=0,
                                                      z=z,
                                                      nz=nz,
                                                      extra_columns={key: t.nz_data[key]
                                                                     for key in t.nz_data.dtype.names if
                                                                     'nz_' in key and key != 'nz_cosmos'})

            elif t.type == 'Compton_y':
                tracer = sacc.tracers.BaseTracer.make('Map',
                                                      'y_{}'.format(i_t - self.ntracers_counts),
                                                      'Compton_y',
                                                      spin=0,
                                                      ell=-1*np.ones(self.nbands),
                                                      beam_ell=-1*np.ones(self.nbands))

            elif t.type == 'kappa':
                tracer = sacc.tracers.BaseTracer.make('Map',
                                                      'kappa_{}'.format(i_t - self.ntracers_counts - self.ntracers_comptony),
                                                      'kappa',
                                                      spin=0,
                                                      ell=-1*np.ones(self.nbands),
                                                      beam_ell=-1*np.ones(self.nbands))

            elif t.type == 'cosmic_shear':
                z = (t.nz_data['z_i'] + t.nz_data['z_f']) * 0.5
                nz = t.nz_data['nz_cosmos']
                tracer = sacc.tracers.BaseTracer.make('NZ',
                                                      'wl_{}'.format(i_t-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                                      'cosmic_shear',
                                                      spin=2,
                                                      z=z,
                                                      nz=nz,
                                                      extra_columns={key: t.nz_data[key]
                                                                     for key in t.nz_data.dtype.names if
                                                                     'nz_' in key and key != 'nz_cosmos'})

            else:
                raise NotImplementedError('Only tracer types delta_g, cosmic_shear, Compton_y supported.')

            sacc_tracers.append(tracer)

        return sacc_tracers

    def write_vector_to_sacc(self, fname_out, sacc_t, cls, ells, windows, covar=None) :
        """
        Write a vector of power spectrum measurements into a SACC file.
        :param fname_out: path to output file
        :param sacc_t: list of SACC tracers
        :param sacc_b: SACC Binning object.
        :param cls: list of power spectrum measurements.
        :param covar: covariance matrix:
        :param verbose: do you want to print out information about the SACC file?
        """

        ells_all = np.arange(self.lmax + 1)

        # Add tracers to sacc
        saccfile = sacc.Sacc()
        for trc in sacc_t:
            saccfile.add_tracer_object(trc)

        map_i = 0
        for tr_i in range(self.ntracers):
            map_j = map_i
            for tr_j in range(tr_i, self.ntracers):
                wins = sacc.Window(ells_all, windows[tr_i][tr_j].T)
                if sacc_t[tr_i].quantity == 'delta_g' and sacc_t[tr_j].quantity == 'delta_g':
                    saccfile.add_ell_cl('cl_00',
                                 'gc_{}'.format(tr_i),
                                 'gc_{}'.format(tr_j),
                                 ells,
                                 cls[map_i, map_j, :],
                                 window=wins,
                                 window_id=range(self.nbands)
                                 )
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'delta_g' and sacc_t[tr_j].quantity == 'cosmic_shear':
                    saccfile.add_ell_cl('cl_0e',
                                 'gc_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 ells,
                                 cls[map_i, map_j, :],
                                 window=wins,
                                 window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_0b',
                                 'gc_{}'.format(tr_i),
                                 'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 ells,
                                 cls[map_i, map_j+1, :],
                                 window=wins,
                                 window_id=range(self.nbands))
                    map_j += 2

                elif sacc_t[tr_i].quantity == 'cosmic_shear' and sacc_t[tr_j].quantity == 'delta_g':
                    saccfile.add_ell_cl('cl_0e',
                                        'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        'gc_{}'.format(tr_j),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_0b',
                                        'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        'gc_{}'.format(tr_j),
                                        ells,
                                        cls[map_i+1, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'Compton_y' and sacc_t[tr_j].quantity == 'Compton_y':
                    saccfile.add_ell_cl('cl_00',
                                        'y_{}'.format(tr_i - self.ntracers_counts),
                                        'y_{}'.format(tr_j - self.ntracers_counts),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands)
                                        )
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'delta_g' and sacc_t[tr_j].quantity == 'Compton_y':
                    saccfile.add_ell_cl('cl_00',
                                        'gc_{}'.format(tr_i),
                                        'y_{}'.format(tr_j - self.ntracers_counts),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'Compton_y' and sacc_t[tr_j].quantity == 'delta_g':
                    saccfile.add_ell_cl('cl_00',
                                        'y_{}'.format(tr_i - self.ntracers_counts),
                                        'gc_{}'.format(tr_j),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'Compton_y' and sacc_t[tr_j].quantity == 'cosmic_shear':
                    saccfile.add_ell_cl('cl_0e',
                                        'y_{}'.format(tr_i - self.ntracers_counts),
                                        'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_0b',
                                        'y_{}'.format(tr_i - self.ntracers_counts),
                                        'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        ells,
                                        cls[map_i, map_j + 1, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 2

                elif sacc_t[tr_i].quantity == 'cosmic_shear' and sacc_t[tr_j].quantity == 'Compton_y':
                    saccfile.add_ell_cl('cl_0e',
                                        'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        'y_{}'.format(tr_j - self.ntracers_counts),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_0b',
                                        'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        'y_{}'.format(tr_j - self.ntracers_counts),
                                        ells,
                                        cls[map_i + 1, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'kappa' and sacc_t[tr_j].quantity == 'kappa':
                    saccfile.add_ell_cl('cl_00',
                                        'kappa_{}'.format(tr_i - self.ntracers_counts - self.ntracers_comptony),
                                        'kappa_{}'.format(tr_j - self.ntracers_counts - self.ntracers_comptony),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands)
                                        )
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'delta_g' and sacc_t[tr_j].quantity == 'kappa':
                    saccfile.add_ell_cl('cl_00',
                                        'gc_{}'.format(tr_i),
                                        'kappa_{}'.format(tr_j - self.ntracers_counts - self.ntracers_comptony),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'kappa' and sacc_t[tr_j].quantity == 'delta_g':
                    saccfile.add_ell_cl('cl_00',
                                        'kappa_{}'.format(tr_i - self.ntracers_counts - self.ntracers_comptony),
                                        'gc_{}'.format(tr_j),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'kappa' and sacc_t[tr_j].quantity == 'cosmic_shear':
                    saccfile.add_ell_cl('cl_0e',
                                        'kappa_{}'.format(tr_i - self.ntracers_counts - self.ntracers_comptony),
                                        'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_0b',
                                        'kappa_{}'.format(tr_i - self.ntracers_counts - self.ntracers_comptony),
                                        'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        ells,
                                        cls[map_i, map_j + 1, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 2

                elif sacc_t[tr_i].quantity == 'cosmic_shear' and sacc_t[tr_j].quantity == 'kappa':
                    saccfile.add_ell_cl('cl_0e',
                                        'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        'kappa_{}'.format(tr_j - self.ntracers_counts - self.ntracers_comptony),
                                        ells,
                                        cls[map_i, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_0b',
                                        'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                        'kappa_{}'.format(tr_j - self.ntracers_counts - self.ntracers_comptony),
                                        ells,
                                        cls[map_i + 1, map_j, :],
                                        window=wins,
                                        window_id=range(self.nbands))
                    map_j += 1

                elif sacc_t[tr_i].quantity == 'cosmic_shear' and sacc_t[tr_j].quantity == 'cosmic_shear':
                    saccfile.add_ell_cl('cl_ee',
                                 'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 ells,
                                 cls[map_i, map_j, :],
                                 window=wins,
                                 window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_eb',
                                 'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 ells,
                                 cls[map_i+1, map_j, :],
                                 window=wins,
                                 window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_be',
                                 'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 ells,
                                 cls[map_i, map_j+1, :],
                                 window=wins,
                                 window_id=range(self.nbands))
                    saccfile.add_ell_cl('cl_bb',
                                 'wl_{}'.format(tr_i-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 'wl_{}'.format(tr_j-self.ntracers_counts -self.ntracers_comptony - self.ntracers_kappa),
                                 ells,
                                 cls[map_i+1, map_j+1, :],
                                 window=wins,
                                 window_id=range(self.nbands))
                    map_j += 2

            if sacc_t[tr_i].spin == 2:
                map_i += 2
            else:
                map_i += 1

        if covar is not None :
            saccfile.add_covariance(covar)

        saccfile.save_fits(fname_out, overwrite=True)

    def convert_sacc_to_clarr(self, saccmean, trc):

        cl_arr = np.zeros((self.nmaps, self.nmaps, self.nbands))

        map_i = 0
        ind_sacc = 0
        for tr_i in range(self.ntracers):
            map_j = map_i
            for tr_j in range(tr_i, self.ntracers):
                if trc[tr_i].spin == 0 and trc[tr_j].spin == 0:
                    cl_arr[map_i, map_j, :] = saccmean[ind_sacc*self.nbands:(ind_sacc+1)*self.nbands]
                    map_j += 1
                    ind_sacc += 1
                elif trc[tr_i].spin == 0 and trc[tr_j].spin == 2:
                    tempmeans = saccmean[ind_sacc*self.nbands:(ind_sacc+2)*self.nbands].reshape((2, -1))
                    cl_arr[map_i, map_j, :] = tempmeans[0, :]
                    cl_arr[map_i, map_j+1, :] = tempmeans[1, :]
                    map_j += 2
                    ind_sacc += 2
                elif trc[tr_i].spin == 2 and trc[tr_j].spin == 0:
                    tempmeans = saccmean[ind_sacc * self.nbands:(ind_sacc+2) * self.nbands].reshape((2, -1))
                    cl_arr[map_i, map_j, :] = tempmeans[0, :]
                    cl_arr[map_i+1, map_j, :] = tempmeans[1, :]
                    map_j += 1
                    ind_sacc += 2
                elif trc[tr_i].spin == 2 and trc[tr_j].spin == 2:
                    tempmeans = saccmean[ind_sacc * self.nbands:(ind_sacc+4) * self.nbands].reshape((4, -1))
                    cl_arr[map_i, map_j, :] = tempmeans[0, :]
                    cl_arr[map_i+1, map_j, :] = tempmeans[1, :]
                    cl_arr[map_i, map_j+1, :] = tempmeans[2, :]
                    cl_arr[map_i+1, map_j+1, :] = tempmeans[3, :]
                    map_j += 2
                    ind_sacc += 4

            if trc[tr_i].spin == 2:
                map_i += 2
            else:
                map_i += 1

        return cl_arr

    def mapping(self, trcs):

        self.pss2tracers = [[0 for i in range(self.nmaps)] for ii in range(self.nmaps)]
        self.maps2tracers = [0 for i in range(self.nmaps)]

        map_i = 0
        for tr_i in range(self.ntracers):
            map_j = map_i
            self.maps2tracers[map_i] = tr_i
            for tr_j in range(tr_i, self.ntracers):
                if trcs[tr_i].spin == 0 and trcs[tr_j].spin == 0:
                    self.pss2tracers[map_i][map_j] = (tr_i, tr_j)
                    if map_i != map_j:
                        self.pss2tracers[map_j][map_i] = (tr_i, tr_j)
                    map_j += 1
                elif trcs[tr_i].spin == 0 and trcs[tr_j].spin == 2:
                    # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                    self.pss2tracers[map_i][map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i][map_j+1] = (tr_i, tr_j)
                    if map_i != map_j:
                        self.pss2tracers[map_j][map_i] = (tr_i, tr_j)
                        self.pss2tracers[map_j+1][map_i] = (tr_i, tr_j)
                    map_j += 2
                elif trcs[tr_i].spin == 2 and trcs[tr_j].spin == 0:
                    # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
                    self.pss2tracers[map_i][map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i+1][map_j] = (tr_i, tr_j)
                    if map_i != map_j:
                        self.pss2tracers[map_j][map_i] = (tr_i, tr_j)
                        self.pss2tracers[map_j][map_i+1] = (tr_i, tr_j)
                    map_j += 1
                else:
                    # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
                    self.pss2tracers[map_i][map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i+1][map_j] = (tr_i, tr_j)
                    self.pss2tracers[map_i][map_j+1] = (tr_i, tr_j)
                    self.pss2tracers[map_i+1][map_j+1] = (tr_i, tr_j)
                    if map_i != map_j:
                        self.pss2tracers[map_j][map_i] = (tr_i, tr_j)
                        self.pss2tracers[map_j][map_i+1] = (tr_i, tr_j)
                        self.pss2tracers[map_j+1][map_i] = (tr_i, tr_j)
                        self.pss2tracers[map_j+1][map_i+1] = (tr_i, tr_j)
                    map_j += 2

            if trcs[tr_i].spin == 2:
                map_i += 2
            else:
                map_i += 1

        tracer_combs = []
        for i1 in range(self.ntracers):
            for j1 in range(i1, self.ntracers):
                tracer_combs.append((i1, j1))

        self.tracers2maps = [[[] for i in range(self.ntracers)] for ii in range(self.ntracers)]

        for trcs in tracer_combs:
            tr_i, tr_j = trcs
            for i in range(len(self.pss2tracers)):
                for ii in range(len(self.pss2tracers[i])):
                    if self.pss2tracers[i][ii] == trcs:
                        self.tracers2maps[tr_i][tr_j].append([i, ii])

        for i in range(len(self.tracers2maps)):
            for ii in range(len(self.tracers2maps[i])):
                self.tracers2maps[i][ii] = np.array(self.tracers2maps[i][ii])
                self.tracers2maps[ii][i] = self.tracers2maps[i][ii]
                
    def run(self) :
        """
        Main function.
        This stage:
        - Produces measurements of the power spectrum with and without contaminant deprojections.
        - Estimates the noise bias
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
        nbands = ell_eff.shape[0]
        self.nbands = nbands
        logger.info('nbands = {}.'.format(self.nbands))

        tracers_nc, tracers_wc = self.get_all_tracers(temps)

        self.ntracers = len(tracers_nc)
        self.nmaps = self.ntracers_counts + self.ntracers_comptony + self.ntracers_kappa + 2*self.ntracers_shear

        logger.info("Translating into SACC tracers.")
        tracers_sacc=self.get_sacc_tracers(tracers_nc)

        # Set up mapping
        self.mapping(tracers_nc)

        logger.info("Getting MCM.")
        wsp = self.get_mcm(tracers_nc,bpws)

        logger.info("Computing window function.")
        windows = self.get_windows(tracers_nc, wsp)

        logger.info("Computing power spectra.")
        logger.info(" No deprojections.")
        cls_wodpj,_=self.get_power_spectra(tracers_nc,wsp,bpws)
        logger.info(" W. deprojections.")
        cls_wdpj,cls_wdpj_coupled=self.get_power_spectra(tracers_wc,wsp,bpws)
        self.ncross = self.nmaps*(self.nmaps + 1)//2 + self.ntracers_shear

        logger.info("Getting guess power spectra.")
        lth,clth=self.get_cl_guess(ell_eff,cls_wdpj)

        logger.info("Computing deprojection bias.")
        cls_wdpj, cl_deproj_bias=self.get_dpj_bias(tracers_wc,lth,clth,cls_wdpj_coupled,wsp,bpws)

        logger.info("Computing noise bias.")
        nls=self.get_noise(tracers_nc,wsp,bpws)

        logger.info("Writing output")
        self.write_vector_to_sacc(self.get_output_fname('noi_bias',ext='sacc'), tracers_sacc,
                                  nls, ell_eff, windows)
        logger.info('Written noise bias.')
        self.write_vector_to_sacc(self.get_output_fname('dpj_bias',ext='sacc'), tracers_sacc,
                                  cl_deproj_bias, ell_eff, windows)
        logger.info('Written deprojection bias.')
        self.write_vector_to_sacc(self.get_output_fname('power_spectra_wodpj',ext='sacc'), tracers_sacc,
                                  cls_wodpj, ell_eff, windows)
        logger.info('Written power spectra without deprojection.')
        self.write_vector_to_sacc(self.get_output_fname('power_spectra_wdpj',ext='sacc'), tracers_sacc,
                                  cls_wdpj, ell_eff, windows)
        logger.info('Written deprojected power spectra.')

if __name__ == '__main__':
    cls = PipelineStage.main()
