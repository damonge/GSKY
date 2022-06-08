#! /usr/bin/env python

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .types import FitsFile,ASCIIFile,BinaryFile,NpzFile,SACCFile,DummyFile
import numpy as np
from operator import add
import multiprocessing
import copy
import pymaster as nmt
from astropy.io import fits
from astropy.table import Table, vstack
from .tracer import Tracer
from .map_utils import (createCountsMap,
                        createMeanStdMaps,
                        createMask,
                        removeDisconnected,
                        createSpin2Map,
                        createW2QU2Map)
import sacc
from scipy.interpolate import interp1d
from .flatmaps import FlatMapInfo,read_flat_map,compare_infos

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# KEYS = ['probes', 'spins', 'nprobes', 'nspin2', 'ncls', 'nautocls']

class CovFromMocks(object):
    """
    Construct covariance matrix from mock catalogs
    """
    def __init__(self):
        self.enrich_params()

    def make_masked_fraction(self, cat, fsk, config, mask_fulldepth=False):
        """
        Produces a masked fraction map
        :param cat: input catalog
        :param fsk: FlatMapInfo object describing the
            geometry of the output map
        """
        logger.info("Generating masked fraction map")

        masked = np.ones(len(cat))
        masked_fraction, _ = createMeanStdMaps(cat[config['ra']],
                                               cat[config['dec']],
                                               masked, fsk)
        masked_fraction_cont = removeDisconnected(masked_fraction, fsk)
        return masked_fraction_cont

    def pz_binning(self, cat, config):
        zi_arr = config['pz_bins'][:-1]
        zf_arr = config['pz_bins'][1:]
        self.nbins = len(zi_arr)
        zs = cat['z_source_mock']

        # Assign all galaxies to bin -1
        bin_number = np.zeros(len(cat), dtype=int) - 1

        for ib, (zi, zf) in enumerate(zip(zi_arr, zf_arr)):
            msk = (zs <= zf) & (zs > zi)
            bin_number[msk] = ib
        return bin_number

    def get_gamma_maps(self, cat, fsk, config):
        """
        Get gamma1, gamma2 maps and corresponding mask from catalog.
        :param cat:
        :return:
        """

        # if 'i_hsmshaperegauss_e1_calib' not in cat.dtype.names:
        #     raise RuntimeError('get_gamma_maps must be called with '
        #                        'calibrated shear catalog. Aborting.')
        maps = []

        # Tomographic maps
        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat['tomo_bin'] >= 0) 
            subcat = cat[msk_bin]
            if config['shape_noise'] == False:
                gammamaps, gammamasks = createSpin2Map(subcat[config['ra']],
                                                       subcat[config['dec']],
                                                       subcat['shear1_sim']/(1-subcat['kappa']),
                                                       subcat['shear2_sim']/(1-subcat['kappa']), fsk,
                                                       weights=subcat['weight'],
                                                       shearrot=config['shearrot'])
            else:
                gammamaps, gammamasks = createSpin2Map(subcat[config['ra']],
                                                       subcat[config['dec']],
                                                       subcat['e1_mock'],
                                                       subcat['e2_mock'], fsk,
                                                       weights=subcat['weight'],
                                                       shearrot=config['shearrot'])
            maps_combined = [gammamaps, gammamasks]
            maps.append(maps_combined)

        return maps

    def get_e2rms(self, cat, config):
        """
        Get e1_2rms, e2_2rms from catalog.
        :param cat:
        :return:
        """

        # if 'i_hsmshaperegauss_e1_calib' not in cat.dtype.names:
        #     raise RuntimeError('get_e2rms must be called with '
        #                        'calibrated shear catalog. Aborting.')
        e2rms_arr = []

        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat['tomo_bin'] >= 0)
            subcat = cat[msk_bin]
            if config['shape_noise'] == False:
                e1_2rms = np.average((subcat['shear1_sim']/(1-subcat['kappa']))**2,
                                     weights=subcat['weight'])
                e2_2rms = np.average((subcat['shear2_sim']/(1-subcat['kappa']))**2,
                                     weights=subcat['weight'])
            else:
                e1_2rms = np.average((subcat['e1_mock'])**2,
                                     weights=subcat['weight'])
                e2_2rms = np.average((subcat['e2_mock'])**2,
                                     weights=subcat['weight'])

            e2rms_combined = np.array([e1_2rms, e2_2rms])
            e2rms_arr.append(e2rms_combined)

        return np.array(e2rms_arr)

    def get_w2e2(self, cat, fsk, config, return_maps=False):
        """
        Compute the weighted mean squared ellipticity in a pixel, averaged over the whole map (used for analytic shape
        noise estimation).
        :param cat:
        :return:
        """

        # if 'i_hsmshaperegauss_e1_calib' not in cat.dtype.names:
        #     raise RuntimeError('get_gamma_maps must be called with '
        #                        'calibrated shear catalog. Aborting.')
        w2e2 = []
        w2e2maps = []

        for ibin in self.bin_indxs:
            if ibin != -1:
                # msk_bin = (cat['tomo_bin'] == ibin) & cat['shear_cat']
                msk_bin = (cat['tomo_bin'] == ibin)
            else:
                # msk_bin = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                msk_bin = (cat['tomo_bin'] >= 0)
            subcat = cat[msk_bin]
            if config['shape_noise'] == False:
                w2e2maps_curr = createW2QU2Map(subcat[config['ra']],
                                                       subcat[config['dec']],
                                                       subcat['shear1_sim']/(1-subcat['kappa']),
                                                       subcat['shear2_sim']/(1-subcat['kappa']), fsk,
                                                       weights=subcat['weight'])
            else:
                w2e2maps_curr = createW2QU2Map(subcat[config['ra']],
                                                       subcat[config['dec']],
                                                       subcat['e1_mock'],
                                                       subcat['e2_mock'], fsk,
                                                       weights=subcat['weight'])

            w2e2_curr = 0.5*(np.mean(w2e2maps_curr[0]) + np.mean(w2e2maps_curr[1]))
            w2e2.append(w2e2_curr)
            w2e2maps.append(w2e2maps_curr)

        if not return_maps:
            return np.array(w2e2)
        else:
            return np.array(w2e2), w2e2maps

    def enrich_params(self):
        """
        Infers the unspecified parameters from the parameters provided and
        updates the parameter dictionary accordingly.
        :param :
        :return :
        """
        PARAMS_KEYS = ['nprobes', 'ncls', 'l0_bins', 'lf_bins', 'nell',
                  'nspin2', 'nautocls']
        default_key_vals = [4, 10, [], [], 0,
                  1, 4]
        self.params = {}
        self.params['nprobes'] = 4
        self.params['ncls'] = int(self.params['nprobes']*(self.params['nprobes'] + 1.)/2.)
        ell_bpws = [100.0,200.0,300.0,400.0,600.0,800.0,1000.0,1400.0,1800.0,2200.0,3000.0,3800.0,4600.0,6200.0,7800.0,9400.0,12600.0,15800.0]
        self.params['l0_bins'] = np.array(ell_bpws)[:-1]
        self.params['lf_bins'] = np.array(ell_bpws)[1:]
        self.params['nell'] = int(self.params['l0_bins'].shape[0])
        self.params['nspin2'] = 1
        self.params['nautocls'] = self.params['nprobes']+self.params['nspin2']

        if not hasattr(self, 'wsps'):
            logger.info('Applying workspace caching.')
            logger.info('Setting up workspace attribute.')
            self.wsps = [[None for i in range(self.params['nprobes'])] for ii in range(self.params['nprobes'])]

    def go(self):
        config={'plots_dir': None,
          'min_snr': 10., 'depth_cut': 24.5,
          'mapping': {'wcs': None, 'res': 0.0285,
                      'res_bo': 0.003, 'pad': 0.1,
                      'projection': 'CAR'},
          'band': 'i', 'depth_method': 'fluxerr',
          'shearrot': 'noflip', 'mask_type': 'sirius',
          'ra':  'ra_mock', 'dec':  'dec_mock',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shearrot': 'noflip',
          'ra':  'ra_mock', 'dec':  'dec_mock', 'shape_noise': True,
          'mocks_dir': '/projects/HSC/weaklens/xlshare/S19ACatalogs/catalog_mock/fields/XMM/'}

        n_realizations = len(os.listdir(config['mocks_dir']))
        n_realizations = 8
        realizations = np.arange(n_realizations)
        ncpus = multiprocessing.cpu_count()
        ncpus = 4
        # ncpus = 1
        logger.info('Number of realizations {}.'.format(n_realizations))
        logger.info('Number of available CPUs {}.'.format(ncpus))
        pool = multiprocessing.Pool(processes = ncpus)

        # Pool map preserves the call order!
        reslist = pool.map(self, realizations, chunksize=int(realizations.shape[0]/ncpus))

        logger.info('done')
        pool.close() # no more tasks
        pool.join()  # wrap up current tasks

    def __call__(self, realization):
        config={'plots_dir': None,
          'min_snr': 10., 'depth_cut': 24.5,
          'mapping': {'wcs': None, 'res': 0.0285,
                      'res_bo': 0.003, 'pad': 0.1,
                      'projection': 'CAR'},
          'band': 'i', 'depth_method': 'fluxerr',
          'shearrot': 'noflip', 'mask_type': 'sirius',
          'ra':  'ra_mock', 'dec':  'dec_mock',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shearrot': 'noflip',
          'ra':  'ra_mock', 'dec':  'dec_mock', 'shape_noise': True,
          'mocks_dir': '/projects/HSC/weaklens/xlshare/S19ACatalogs/catalog_mock/fields/XMM/'}
        logger.info('Running realization : {}.'.format(realization))
        band = config['band']
        self.mpp = config['mapping']
        #map realization number to mock catalog
        r_num = int(np.floor(realization/13))
        r_num_str = format(r_num, '03d')
        rotmat = int(realization - (r_num*13))
        name = 'mock_nres13_r'+r_num_str+'_rotmat'+str(rotmat)+'_shear_catalog.fits'
        # Read catalog
        logger.info('reading catalog')
        cat = Table.read(config['mocks_dir']+name)
        #ReduceCat
        if 'VVDS' in config['mocks_dir']:
            logger.info("Shifting RA by -30 degrees for VVDS")
            change_in_ra = -30.0
            init_ra_vals = cat[config['ra']].copy()
            cat[config['ra']] = init_ra_vals+(np.ones(len(init_ra_vals))*change_in_ra)
            cat[config['ra']][cat[config['ra']]<0] += 360.0
        logger.info('generating masked fraction')
        fsk = FlatMapInfo.from_coords(cat[config['ra']],
                              cat[config['dec']],
                              self.mpp)
        masked_fraction_cont = self.make_masked_fraction(cat, fsk, config,
                                                 mask_fulldepth=True)
        logger.info('tomographic binning')
        cat['tomo_bin'] = self.pz_binning(cat, config)
        #ShearMapper
        self.nbins = len(config['pz_bins'])-1
        if 'ntomo_bins' in config:
            self.bin_indxs = config['ntomo_bins']
        else:
            self.bin_indxs = range(self.nbins)
        logger.info('getting e2rms')
        e2rms = self.get_e2rms(cat, config)
        logger.info('getting w2e2')
        w2e2 = self.get_w2e2(cat, fsk, config, return_maps=False)
        logger.info('getting gamma maps')
        gammamaps = self.get_gamma_maps(cat, fsk, config)

        b = nmt.NmtBinFlat(self.params['l0_bins'], self.params['lf_bins'])

        masks = []
        weightmask=True
        for i in range(self.params['nprobes']):
            if weightmask ==True:
                logger.info('Using weightmask.')
                print(len(gammamaps))
                mask_temp = gammamaps[i][1][0]
                logger.info(mask_temp)
            else:
                logger.info('Using binary mask.')
                mask_temp = gammamaps[i][1][1]
            mask_temp = np.array(mask_temp).reshape([fsk.ny, fsk.nx])
            masks.append(mask_temp)
        
        self.masks = masks


        # for j in range(self.nbins):
        #     for jj in range(j+1):

        #         probe1 = 'wl_'+str(j)
        #         probe2 = 'wl_'+str(jj)
        #         spin1 = 2
        #         spin2 = 2

        #         logger.info('Computing the power spectrum between probe1 = {} and probe2 = {}.'.format(probe1, probe2))
        #         logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

        #         # Define flat sky spin-2 field
        #         emaps = [gammamaps[j], gammamaps[j+1]]
        #         f2_1 = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly), self.masks[j],
        #                                 emaps, purify_b=False)
        #         # Define flat sky spin-2 field
        #         emaps = [maps[jj], maps[jj+1]]
        #         f2_2 = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly), self.masks[jj],
        #                                 emaps, purify_b=False)

        #         if self.wsps[j][jj] is None:
        #             logger.info('Workspace element for j, jj = {}, {} not set.'.format(j, jj))
        #             logger.info('Computing workspace element.')
        #             wsp = nmt.NmtWorkspaceFlat()
        #             wsp.compute_coupling_matrix(f2_1, f2_2, b)
        #             self.wsps[j][jj] = wsp
        #             if j != jj:
        #                self.wsps[jj][j] = wsp
        #         else:
        #             logger.info('Workspace element already set for j, jj = {}, {}.'.format(j, jj))

        #         # Compute pseudo-Cls
        #         cl_coupled = nmt.compute_coupled_cell_flat(f2_1, f2_2, b)
        #         # Uncoupling pseudo-Cls
        #         cl_uncoupled = self.wsps[j][jj].decouple_cell(cl_coupled)

        #         # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
        #         tempclse = cl_uncoupled[0]
        #         tempclseb = cl_uncoupled[1]
        #         tempclsb = cl_uncoupled[3]

        #         cls[j, jj, :] = tempclse
        #         cls[j+self.params['nspin2'], jj, :] = tempclseb
        #         cls[j+self.params['nspin2'], jj+self.params['nspin2'], :] = tempclsb


        # # If noise is True, then we need to compute the noise from simulations
        # # We therefore generate different noise maps for each realisation so that
        # # we can then compute the noise power spectrum from these noise realisations
        # if self.params['signal'] and self.params['noise']:
        #     if self.params['add_cs_sig']:
        #         # Determine the noise bias on the auto power spectrum for each realisation
        #         # For the cosmic shear, we now add the shear from the noisefree signal maps to the
        #         # data i.e. we simulate how we would do it in real life
        #         logger.info('Adding cosmic shear signal to noise maps.')
        #         noisemaps = self.noisemaps.generate_maps(signalmaps)
        #     else:
        #         logger.info('Not adding cosmic shear signal to noise maps.')
        #         noisemaps = self.noisemaps.generate_maps()
        #     for j, probe in enumerate(self.params['probes']):
        #         logger.info('Computing the noise power spectrum for {}.'.format(probe))
        #         if self.params['spins'][j] == 2:
        #             # Define flat sky spin-2 field
        #             emaps = [noisemaps[j], noisemaps[j+self.params['nspin2']]]
        #             f2 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
        #                                   emaps, purify_b=False)

        #             if self.wsps[j][j] is None:
        #                 logger.info('Workspace element for j, j = {}, {} not set.'.format(j, j))
        #                 logger.info('Computing workspace element.')
        #                 wsp = nmt.NmtWorkspaceFlat()
        #                 wsp.compute_coupling_matrix(f2, f2, b)
        #                 self.wsps[j][j] = wsp
        #             else:
        #                 logger.info('Workspace element already set for j, j = {}, {}.'.format(j, j))

        #             # Compute pseudo-Cls
        #             cl_coupled = nmt.compute_coupled_cell_flat(f2, f2, b)
        #             # Uncoupling pseudo-Cls
        #             cl_uncoupled = self.wsps[j][j].decouple_cell(cl_coupled)

        #             # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
        #             tempclse = cl_uncoupled[0]
        #             tempclsb = cl_uncoupled[3]

        #             noisecls[j, j, :] = tempclse
        #             noisecls[j+self.params['nspin2'], j+self.params['nspin2'], :] = tempclsb
        #         else:
        #             # Define flat sky spin-0 field
        #             emaps = [noisemaps[j]]
        #             f0 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
        #                                   emaps, purify_b=False)

        #             if self.wsps[j][j] is None:
        #                 logger.info('Workspace element for j, j = {}, {} not set.'.format(j, j))
        #                 logger.info('Computing workspace element.')
        #                 wsp = nmt.NmtWorkspaceFlat()
        #                 wsp.compute_coupling_matrix(f0, f0, b)
        #                 self.wsps[j][j] = wsp
        #             else:
        #                 logger.info('Workspace element already set for j, j = {}, {}.'.format(j, j))

        #             # Compute pseudo-Cls
        #             cl_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
        #             # Uncoupling pseudo-Cls
        #             cl_uncoupled = self.wsps[j][j].decouple_cell(cl_coupled)
        #             noisecls[j, j, :] = cl_uncoupled

        # if not self.params['signal'] and self.params['noise']:
        #     noisecls = copy.deepcopy(cls)
        #     cls = np.zeros_like(noisecls)

        # return cls, noisecls, ells_uncoupled