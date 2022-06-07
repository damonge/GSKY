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
        pass

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

    def get_w2e2(self, cat, config, fsk, return_maps=False):
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
            print(config['shape_noise'])
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
        self.fsk = masked_fraction_cont
        self.nbins = len(config['pz_bins'])-1
        if 'ntomo_bins' in config:
            self.bin_indxs = config['ntomo_bins']
        else:
            self.bin_indxs = range(self.nbins)
        logger.info('getting e2rms')
        e2rms = self.get_e2rms(cat, config)
        logger.info('getting w2e2')
        w2e2 = self.get_w2e2(cat, self.fsk, config, return_maps=False)
        logger.info('getting gamma maps')
        gammamaps = self.get_gamma_maps(cat, self.fsk, config)

    # def __init__(self, masks, simparams={}, noiseparams={}):
    #     """
    #     Constructor for the MockSurvey class
    #     """

    #     self.params = simparams
    #     self.enrich_params()
    #     self.masks = masks

    #     logger.info("Reading masked fraction from {}.".format(noiseparams['path2fsk']))
    #     self.fsk, _ = read_flat_map(noiseparams['path2fsk'])

    #     if simparams['theory_sacc'] != 'NONE':
    #         logger.info('theory_sacc provided. Generating signal realizations.')
    #         self.params['signal'] = True
    #         self.simmaps = SimulatedMaps(self.fsk, simparams)
    #     else:
    #         logger.info('theory_sacc is NONE. Not generating signal realizations.')
    #         self.params['signal'] = False
    #     if noiseparams != {}:
    #         logger.info('Generating noise realizations.')
    #         # Need to generate noise realisations as well
    #         self.params['noise'] = True
    #         noiseparams = self.enrich_noise_params(noiseparams)
    #         self.noisemaps = NoiseMaps(noiseparams)
    #     else:
    #         logger.info('Not generating noise realizations.')
    #         self.params['noise'] = False

    #     self.print_params()

    # def print_params(self):
    #     """
    #     Prints the parameter combination chosen to initialise MockSurvey.
    #     """

    #     logger.info('SimulatedMaps has been initialised with the following attributes:')
    #     for key in self.params.keys():
    #         print('{} = {}'.format(key, self.params[key]))

    # def enrich_params(self):
    #     """
    #     Infers the unspecified parameters from the parameters provided and
    #     updates the parameter dictionary accordingly.
    #     :param :
    #     :return :
    #     """

    #     self.params['nprobes'] = len(self.params['probes'])
    #     self.params['ncls'] = int(self.params['nprobes']*(self.params['nprobes'] + 1.)/2.)
    #     if 'lmax' in self.params:
    #         self.params['nell'] = self.params['lmax']+1
    #     elif 'ell_bpws'in self.params:
    #         self.params['l0_bins'] = np.array(self.params['ell_bpws'])[:-1]
    #         self.params['lf_bins'] = np.array(self.params['ell_bpws'])[1:]
    #         self.params['nell'] = int(self.params['l0_bins'].shape[0])
    #     self.params['nspin2'] = np.sum(self.params['spins'] == 2).astype('int')
    #     self.params['nautocls'] = self.params['nprobes']+self.params['nspin2']

    #     if not hasattr(self, 'wsps'):
    #         logger.info('Applying workspace caching.')
    #         logger.info('Setting up workspace attribute.')
    #         self.wsps = [[None for i in range(self.params['nprobes'])] for ii in range(self.params['nprobes'])]

    # def enrich_noise_params(self, noiseparams):
    #     """
    #     Infers the unspecified parameters from the parameters provided and
    #     updates the parameter dictionary accordingly.
    #     :param :
    #     :return :
    #     """

    #     for key in KEYS:
    #         noiseparams[key] = self.params[key]

    #     return noiseparams

    # def reconstruct_cls_parallel(self):
    #     """
    #     Calculates the power spectra for different surveys from Gaussian
    #     realisations of input power spectra. Depending on the choices, this
    #     creates mocks of multi-probe surveys taking all the cross-correlations
    #     into account.
    #     :return cls: 4D array of cls for all the realisations and all the probes;
    #     0. and 1. axis denote the power spectrum, 2. axis denotes the realisation number
    #     and the 3. axis gives the cls belonging to this configuration
    #     :return tempells: array of ell values which is equal for all the probes
    #     """

    #     realisations = np.arange(self.params['nrealiz'])
    #     ncpus = multiprocessing.cpu_count()
    #     ncpus = 4
    #     # ncpus = 1
    #     logger.info('Number of available CPUs {}.'.format(ncpus))
    #     pool = multiprocessing.Pool(processes = ncpus)

    #     # Pool map preserves the call order!
    #     reslist = pool.map(self, realisations, chunksize=int(realisations.shape[0]/ncpus))

    #     pool.close() # no more tasks
    #     pool.join()  # wrap up current tasks

    #     # cls, noisecls, tempells = self(realisations)

    #     logger.info('Cls calculation done.')

    #     # Concatenate the cl lists into 4D arrays. The arrays are expanded and concatenated along the
    #     # 2nd axis
    #     cls = np.concatenate([res[0][..., np.newaxis,:] for res in reslist], axis=2)
    #     noisecls = np.concatenate([res[1][..., np.newaxis,:] for res in reslist], axis=2)
    #     tempells = reslist[0][2]

    #     # Compute all workspaces
    #     wsps = self.compute_wsps()

    #     # Remove the noise bias from the auto power spectra
    #     if self.params['signal'] and self.params['noise']:
    #         logger.info('Removing noise bias.')
    #         cls = self.remove_noise(cls, noisecls)

    #     return cls, noisecls, tempells, wsps

    # def remove_noise(self, cls, noisecls):
    #     """
    #     If the mocks are generated with noise, this removes the approximate noise power spectra
    #     from the signal + noise cls.
    #     :param cls: 4D array of signal + noise cls
    #     :param noisecls: 4D array of noise cls
    #     :return cls: 4D array of noise removed cls
    #     """

    #     for i in range(self.params['nautocls']):
    #         cls[i, i, :, :] -= np.mean(noisecls[i,i,:,:], axis=0)

    #     return cls

    # def __call__(self, realiz):
    #     """
    #     Convenience method for calculating the signal and noise cls for
    #     a given mock realization. This is a function that can be pickled and can be thus
    #     used when running the mock generation in parallel using multiprocessing pool.
    #     :param realis: number of the realisation to run
    #     :param noise: boolean flag indicating if noise is added to the mocks
    #     noise=True: add noise to the mocks
    #     noise=False: do not add noise to the mocks
    #     :param probes: list of desired probes to run the mock for
    #     :param maskmat: matrix with the relevant masks for the probes
    #     :param clparams: list of dictionaries with the parameters for calculating the
    #     power spectra for each probe
    #     :return cls: 3D array of signal and noise cls for the given realisation,
    #     0. and 1. axis denote the power spectrum, 2. axis gives the cls belonging
    #     to this configuration
    #     :return noisecls: 3D array of noise cls for the given realisation,
    #     0. and 1. axis denote the power spectrum, 2. axis gives the cls belonging
    #     to this configuration
    #     :return tempells: array of the ell range of the power spectra
    #     """

    #     logger.info('Running realization : {}.'.format(realiz))

    #     cls = np.zeros((self.params['nautocls'], self.params['nautocls'], self.params['nell']))
    #     noisecls = np.zeros_like(cls)

    #     if self.params['signal']:
    #         signalmaps = self.simmaps.generate_maps()
    #     if self.params['noise']:
    #         # We need to add noise maps to the signal maps
    #         noisemaps = self.noisemaps.generate_maps()

    #     if self.params['signal'] and self.params['noise']:
    #         # We need to add the two lists elementwise
    #         maps = list(map(add, signalmaps, noisemaps))
    #     elif self.params['signal'] and not self.params['noise']:
    #         maps = copy.deepcopy(signalmaps)
    #     elif not self.params['signal'] and self.params['noise']:
    #         maps = copy.deepcopy(noisemaps)
    #     else:
    #         raise RuntimeError('Either signal or noise must be True. Aborting.')

    #     b = nmt.NmtBinFlat(self.params['l0_bins'], self.params['lf_bins'])
    #     # The effective sampling rate for these bandpowers can be obtained calling:
    #     ells_uncoupled = b.get_effective_ells()

    #     # First compute the cls of map realization for all the probes
    #     for j in range(self.params['nprobes']):
    #         for jj in range(j+1):

    #             if j == jj:
    #                 compute_cls = True
    #             else:
    #                 if self.params['signal']:
    #                     compute_cls = True
    #                 else:
    #                     compute_cls = False

    #             if compute_cls:
    #                 probe1 = self.params['probes'][j]
    #                 probe2 = self.params['probes'][jj]
    #                 spin1 = self.params['spins'][j]
    #                 spin2 = self.params['spins'][jj]

    #                 logger.info('Computing the power spectrum between probe1 = {} and probe2 = {}.'.format(probe1, probe2))
    #                 logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))
    #                 if spin1 == 2 and spin2 == 0:
    #                     # Define flat sky spin-2 field
    #                     emaps = [maps[j], maps[j+self.params['nspin2']]]
    #                     f2_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                             emaps, purify_b=False)
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[jj]]
    #                     f0_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[jj],
    #                                             emaps, purify_b=False)

    #                     if self.wsps[j][jj] is None:
    #                         logger.info('Workspace element for j, jj = {}, {} not set.'.format(j, jj))
    #                         logger.info('Computing workspace element.')
    #                         wsp = nmt.NmtWorkspaceFlat()
    #                         wsp.compute_coupling_matrix(f2_1, f0_1, b)
    #                         self.wsps[j][jj] = wsp
    #                         if j != jj:
    #                            self.wsps[jj][j] = wsp
    #                     else:
    #                         logger.info('Workspace element already set for j, jj = {}, {}.'.format(j, jj))

    #                     # Compute pseudo-Cls
    #                     cl_coupled = nmt.compute_coupled_cell_flat(f2_1, f0_1, b)
    #                     # Uncoupling pseudo-Cls
    #                     cl_uncoupled = self.wsps[j][jj].decouple_cell(cl_coupled)

    #                     # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
    #                     tempclse = cl_uncoupled[0]
    #                     tempclsb = cl_uncoupled[1]

    #                     cls[j, jj, :] = tempclse
    #                     cls[j+self.params['nspin2'], jj, :] = tempclsb

    #                 elif spin1 == 2 and spin2 == 2:
    #                     # Define flat sky spin-2 field
    #                     emaps = [maps[j], maps[j+self.params['nspin2']]]
    #                     f2_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                             emaps, purify_b=False)
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[jj], maps[jj+self.params['nspin2']]]
    #                     f2_2 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[jj],
    #                                             emaps, purify_b=False)

    #                     if self.wsps[j][jj] is None:
    #                         logger.info('Workspace element for j, jj = {}, {} not set.'.format(j, jj))
    #                         logger.info('Computing workspace element.')
    #                         wsp = nmt.NmtWorkspaceFlat()
    #                         wsp.compute_coupling_matrix(f2_1, f2_2, b)
    #                         self.wsps[j][jj] = wsp
    #                         if j != jj:
    #                            self.wsps[jj][j] = wsp
    #                     else:
    #                         logger.info('Workspace element already set for j, jj = {}, {}.'.format(j, jj))

    #                     # Compute pseudo-Cls
    #                     cl_coupled = nmt.compute_coupled_cell_flat(f2_1, f2_2, b)
    #                     # Uncoupling pseudo-Cls
    #                     cl_uncoupled = self.wsps[j][jj].decouple_cell(cl_coupled)

    #                     # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
    #                     tempclse = cl_uncoupled[0]
    #                     tempclseb = cl_uncoupled[1]
    #                     tempclsb = cl_uncoupled[3]

    #                     cls[j, jj, :] = tempclse
    #                     cls[j+self.params['nspin2'], jj, :] = tempclseb
    #                     cls[j+self.params['nspin2'], jj+self.params['nspin2'], :] = tempclsb

    #                 else:
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[j]]
    #                     f0_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                             emaps, purify_b=False)
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[jj]]
    #                     f0_2 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[jj],
    #                                             emaps, purify_b=False)

    #                     if self.wsps[j][jj] is None:
    #                         logger.info('Workspace element for j, jj = {}, {} not set.'.format(j, jj))
    #                         logger.info('Computing workspace element.')
    #                         wsp = nmt.NmtWorkspaceFlat()
    #                         wsp.compute_coupling_matrix(f0_1, f0_2, b)
    #                         self.wsps[j][jj] = wsp
    #                         if j != jj:
    #                            self.wsps[jj][j] = wsp
    #                     else:
    #                         logger.info('Workspace element already set for j, jj = {}, {}.'.format(j, jj))

    #                     # Compute pseudo-Cls
    #                     cl_coupled = nmt.compute_coupled_cell_flat(f0_1, f0_2, b)
    #                     # Uncoupling pseudo-Cls
    #                     cl_uncoupled = self.wsps[j][jj].decouple_cell(cl_coupled)
    #                     cls[j, jj, :] = cl_uncoupled

    #     # If noise is True, then we need to compute the noise from simulations
    #     # We therefore generate different noise maps for each realisation so that
    #     # we can then compute the noise power spectrum from these noise realisations
    #     if self.params['signal'] and self.params['noise']:
    #         if self.params['add_cs_sig']:
    #             # Determine the noise bias on the auto power spectrum for each realisation
    #             # For the cosmic shear, we now add the shear from the noisefree signal maps to the
    #             # data i.e. we simulate how we would do it in real life
    #             logger.info('Adding cosmic shear signal to noise maps.')
    #             noisemaps = self.noisemaps.generate_maps(signalmaps)
    #         else:
    #             logger.info('Not adding cosmic shear signal to noise maps.')
    #             noisemaps = self.noisemaps.generate_maps()
    #         for j, probe in enumerate(self.params['probes']):
    #             logger.info('Computing the noise power spectrum for {}.'.format(probe))
    #             if self.params['spins'][j] == 2:
    #                 # Define flat sky spin-2 field
    #                 emaps = [noisemaps[j], noisemaps[j+self.params['nspin2']]]
    #                 f2 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                       emaps, purify_b=False)

    #                 if self.wsps[j][j] is None:
    #                     logger.info('Workspace element for j, j = {}, {} not set.'.format(j, j))
    #                     logger.info('Computing workspace element.')
    #                     wsp = nmt.NmtWorkspaceFlat()
    #                     wsp.compute_coupling_matrix(f2, f2, b)
    #                     self.wsps[j][j] = wsp
    #                 else:
    #                     logger.info('Workspace element already set for j, j = {}, {}.'.format(j, j))

    #                 # Compute pseudo-Cls
    #                 cl_coupled = nmt.compute_coupled_cell_flat(f2, f2, b)
    #                 # Uncoupling pseudo-Cls
    #                 cl_uncoupled = self.wsps[j][j].decouple_cell(cl_coupled)

    #                 # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
    #                 tempclse = cl_uncoupled[0]
    #                 tempclsb = cl_uncoupled[3]

    #                 noisecls[j, j, :] = tempclse
    #                 noisecls[j+self.params['nspin2'], j+self.params['nspin2'], :] = tempclsb
    #             else:
    #                 # Define flat sky spin-0 field
    #                 emaps = [noisemaps[j]]
    #                 f0 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                       emaps, purify_b=False)

    #                 if self.wsps[j][j] is None:
    #                     logger.info('Workspace element for j, j = {}, {} not set.'.format(j, j))
    #                     logger.info('Computing workspace element.')
    #                     wsp = nmt.NmtWorkspaceFlat()
    #                     wsp.compute_coupling_matrix(f0, f0, b)
    #                     self.wsps[j][j] = wsp
    #                 else:
    #                     logger.info('Workspace element already set for j, j = {}, {}.'.format(j, j))

    #                 # Compute pseudo-Cls
    #                 cl_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
    #                 # Uncoupling pseudo-Cls
    #                 cl_uncoupled = self.wsps[j][j].decouple_cell(cl_coupled)
    #                 noisecls[j, j, :] = cl_uncoupled

    #     if not self.params['signal'] and self.params['noise']:
    #         noisecls = copy.deepcopy(cls)
    #         cls = np.zeros_like(noisecls)

    #     return cls, noisecls, ells_uncoupled

    # def compute_wsps(self):
    #     """
    #     Convenience method for calculating the NaMaster workspaces for all the probes in the simulation.
    #     :return wsps: wsps list
    #     """

    #     self.wsps = [[None for i in range(self.params['nprobes'])] for ii in range(self.params['nprobes'])]

    #     if self.params['signal']:
    #         signalmaps = self.simmaps.generate_maps()
    #     if self.params['noise']:
    #         # We need to add noise maps to the signal maps
    #         noisemaps = self.noisemaps.generate_maps()

    #     if self.params['signal']:
    #         maps = copy.deepcopy(signalmaps)
    #     elif self.params['noise']:
    #         maps = copy.deepcopy(noisemaps)
    #     else:
    #         raise RuntimeError('Either signal or noise must be True. Aborting.')

    #     b = nmt.NmtBinFlat(self.params['l0_bins'], self.params['lf_bins'])

    #     # Compute workspaces for all the probes
    #     for j in range(self.params['nprobes']):
    #         for jj in range(j+1):

    #             if j == jj:
    #                 compute_cls = True
    #             else:
    #                 if self.params['signal']:
    #                     compute_cls = True
    #                 else:
    #                     compute_cls = False

    #             if compute_cls:
    #                 spin1 = self.params['spins'][j]
    #                 spin2 = self.params['spins'][jj]

    #                 logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))
    #                 if spin1 == 2 and spin2 == 0:
    #                     # Define flat sky spin-2 field
    #                     emaps = [maps[j], maps[j+self.params['nspin2']]]
    #                     f2_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                             emaps, purify_b=False)
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[jj]]
    #                     f0_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[jj],
    #                                             emaps, purify_b=False)

    #                     logger.info('Computing workspace element.')
    #                     wsp = nmt.NmtWorkspaceFlat()
    #                     wsp.compute_coupling_matrix(f2_1, f0_1, b)
    #                     self.wsps[j][jj] = wsp
    #                     if j != jj:
    #                         self.wsps[jj][j] = wsp

    #                 elif spin1 == 2 and spin2 == 2:
    #                     # Define flat sky spin-2 field
    #                     emaps = [maps[j], maps[j+self.params['nspin2']]]
    #                     f2_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                             emaps, purify_b=False)
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[jj], maps[jj+self.params['nspin2']]]
    #                     f2_2 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[jj],
    #                                             emaps, purify_b=False)

    #                     logger.info('Computing workspace element.')
    #                     wsp = nmt.NmtWorkspaceFlat()
    #                     wsp.compute_coupling_matrix(f2_1, f2_2, b)
    #                     self.wsps[j][jj] = wsp
    #                     if j != jj:
    #                         self.wsps[jj][j] = wsp

    #                 else:
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[j]]
    #                     f0_1 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[j],
    #                                             emaps, purify_b=False)
    #                     # Define flat sky spin-0 field
    #                     emaps = [maps[jj]]
    #                     f0_2 = nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), self.masks[jj],
    #                                             emaps, purify_b=False)

    #                     logger.info('Computing workspace element.')
    #                     wsp = nmt.NmtWorkspaceFlat()
    #                     wsp.compute_coupling_matrix(f0_1, f0_2, b)
    #                     self.wsps[j][jj] = wsp
    #                     if j != jj:
    #                         self.wsps[jj][j] = wsp

    #     return self.wsps










