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

class NoiseBiasFromMocks(object):
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

    def randomize_shear_cat(self, cat):
        """
        Rotates each galaxy ellipticity from the galaxy catalog data by a random angle to
        eliminate correlations between galaxy shapes.
        This is used to estimate the shape noise contribution to the shear power spectrum.
        :param cat: structured array with galaxy catalog to randomise
        :return randomiseddata: structured array with galaxy catalog with randomised ellipticities
        """

        logger.info('Randomizing star catalogue.')

        # Copy the input data so it does not get overwritten
        randomizedcat = copy.deepcopy(cat)

        # Seed the random number generator
        np.random.seed(seed=None)

        Mxx = cat['i_sdssshape_psf_shape11']
        Myy = cat['i_sdssshape_psf_shape22']
        Mxy = cat['i_sdssshape_psf_shape12']
        T_I = Mxx + Myy
        e_plus_psf = (Mxx - Myy)/T_I
        e_cross_psf = 2*Mxy/T_I

        M40 = cat['model_moment40']
        M31 = cat['model_moment31']
        M22 = cat['model_moment22']
        M13 = cat['model_moment13']
        M04 = cat['model_moment04']
        M4_plus_psf = M40-M04
        M4_cross_psf = -2*(M13+M31)

        Mxx = cat['i_sdssshape_shape11']
        Myy = cat['i_sdssshape_shape22']
        Mxy = cat['i_sdssshape_shape12']
        T_I = Mxx + Myy
        e_plus_I = (Mxx - Myy)/T_I
        e_cross_I = 2*Mxy/T_I

        M40 = cat['star_moment40']
        M31 = cat['star_moment31']
        M22 = cat['star_moment22']
        M13 = cat['star_moment13']
        M04 = cat['star_moment04']
        M4_plus_I = M40-M04
        M4_cross_I = -2*(M13+M31)


        thetarot = 2.*np.pi*np.random.random_sample((cat['i_sdssshape_psf_shape11'].shape[0], ))

        randomizedcat['e1_psf_random'] = np.cos(2*thetarot)*e_plus_psf - \
                                                 np.sin(2*thetarot)*e_cross_psf

        randomizedcat['e2_psf_random'] = np.sin(2*thetarot)*e_plus_psf + \
                                                 np.cos(2*thetarot)*e_cross_psf

        randomizedcat['m41_psf_random'] = np.cos(2*thetarot)*M4_plus_psf - \
                                                 np.sin(2*thetarot)*M4_cross_psf

        randomizedcat['m42_psf_random'] = np.sin(2*thetarot)*M4_plus_psf + \
                                                 np.cos(2*thetarot)*M4_cross_psf

        randomizedcat['e1_source_random'] = np.cos(2*thetarot)*e_plus_I - \
                                                 np.sin(2*thetarot)*e_cross_I

        randomizedcat['e2_source_random'] = np.sin(2*thetarot)*e_plus_I + \
                                                 np.cos(2*thetarot)*e_cross_I

        randomizedcat['m41_source_random'] = np.cos(2*thetarot)*M4_plus_I - \
                                                 np.sin(2*thetarot)*M4_cross_I

        randomizedcat['m42_source_random'] = np.sin(2*thetarot)*M4_plus_I + \
                                                 np.cos(2*thetarot)*M4_cross_I

        return randomizedcat
    
    def make_PSF_maps(self, star_cat, fsk):
        """
        Get e_PSF, 1, e_PSF, 2, T_PSF maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # PSF of stars
        # Mxx = star_cat['i_sdssshape_psf_shape11']
        # Myy = star_cat['i_sdssshape_psf_shape22']
        # Mxy = star_cat['i_sdssshape_psf_shape12']
        # T_I = Mxx + Myy
        # e_plus_I = (Mxx - Myy)/T_I
        # e_cross_I = 2*Mxy/T_I
        ePSFmaps, ePSFmasks = createSpin2Map(star_cat[config['ra']],
                                             star_cat[config['dec']],
                                             star_cat['e1_psf_random'], star_cat['e2_psf_random'], fsk,
                                             shearrot=config['shearrot'])

        maps = [ePSFmaps, ePSFmasks]

        return maps

    def make_PSF_fourth_moment_maps(self, star_cat, fsk):
        """
        Get fourth moment PSF maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # M40 = star_cat['model_moment40']
        # M31 = star_cat['model_moment31']
        # M22 = star_cat['model_moment22']
        # M13 = star_cat['model_moment13']
        # M04 = star_cat['model_moment04']

        # M4_plus_PSF = M40-M04
        # M4_cross_PSF = -2*(M13+M31)
        M4_PSFmaps, M4_PSFmasks = createSpin2Map(star_cat[config['ra']],
                                             star_cat[config['dec']],
                                             star_cat['m41_psf_random'], star_cat['m42_psf_random'], fsk,
                                             shearrot=config['shearrot'])

        maps = [M4_PSFmaps, M4_PSFmasks]

        return maps

    def make_PSF_res_maps(self, star_cat, fsk):
        """
        Get e_PSF, 1, e_PSF, 2, T_PSF residual maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # PSF of stars
        # Mxx = star_cat['i_sdssshape_psf_shape11']
        # Myy = star_cat['i_sdssshape_psf_shape22']
        # Mxy = star_cat['i_sdssshape_psf_shape12']
        # T_PSF = Mxx + Myy
        # e_plus_PSF = (Mxx - Myy)/T_PSF
        # e_cross_PSF = 2*Mxy/T_PSF

        # Mxx = star_cat['i_sdssshape_shape11']
        # Myy = star_cat['i_sdssshape_shape22']
        # Mxy = star_cat['i_sdssshape_shape12']
        # T_I = Mxx + Myy
        # e_plus_I = (Mxx - Myy)/T_I
        # e_cross_I = 2*Mxy/T_I

        delta_e_plus = star_cat['e1_psf_random'] - star_cat['e1_source_random']
        delta_e_cross = star_cat['e2_psf_random'] - star_cat['e2_source_random']

        ePSFresmaps, ePSFresmasks = createSpin2Map(star_cat[sconfig['ra']],
                                                   star_cat[config['dec']],
                                                   delta_e_plus, delta_e_cross, fsk,
                                                   shearrot=config['shearrot'])

        maps = [ePSFresmaps, ePSFresmasks]

        return maps

    def make_PSF_res_fourth_moment_maps(self, star_cat, fsk):
        """
        Get fourth moment residual maps from catalog.
        Here we go from weighted moments to ellipticities following
        Hirata & Seljak, 2003, arXiv:0301054
        :param cat:
        :return:
        """

        # M40 = star_cat['star_moment40']
        # M31 = star_cat['star_moment31']
        # M22 = star_cat['star_moment22']
        # M13 = star_cat['star_moment13']
        # M04 = star_cat['star_moment04']

        # M4_plus_I = M40-M04
        # M4_cross_I = -2*(M13+M31)

        # M40 = star_cat['model_moment40']
        # M31 = star_cat['model_moment31']
        # M22 = star_cat['model_moment22']
        # M13 = star_cat['model_moment13']
        # M04 = star_cat['model_moment04']

        # M4_plus_PSF = M40-M04
        # M4_cross_PSF = -2*(M13+M31)

        delta_M4_plus = star_cat['m41_psf_random'] - star_cat['m41_source_random']
        delta_M4_cross = star_cat['m42_psf_random'] - star_cat['m42_source_random']

        M4_PSFresmaps, M4_PSFresmasks = createSpin2Map(star_cat[config['ra']],
                                                   star_cat[config['dec']],
                                                   delta_M4_plus, delta_M4_cross, fsk,
                                                   shearrot=config['shearrot'])

        maps = [M4_PSFresmaps, M4_PSFresmasks]

        return maps

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
        self.params['nspin2'] = self.params['nprobes']
        self.params['nautocls'] = self.params['nprobes']+self.params['nspin2']

        if not hasattr(self, 'wsps'):
            logger.info('Applying workspace caching.')
            logger.info('Setting up workspace attribute.')
            self.wsps = [[None for i in range(1)] for ii in range(4)]

    def go(self):
        config={'plots_dir': None,
          'min_snr': 10., 'depth_cut': 24.5,
          'mapping': {'wcs': None, 'res': 0.01666666666667,
                      'res_bo': 0.003, 'pad': 0.2,
                      'projection': 'CAR'},
          'band': 'i', 'depth_method': 'fluxerr',
          'shearrot': 'noflip', 'mask_type': 'sirius',
          'ra':  'i_ra', 'dec':  'i_dec',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shape_noise': True,
          'rm_gama09h_region': True,
          'star_catalog': '/projects/HSC/weaklens/s19a_shape_catalog/star_catalog_higher_moments/XMM_psf.csv'}

        n_realizations = 1000
        # n_realizations = 1
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

        cls = np.concatenate([res[0][..., np.newaxis,:] for res in reslist], axis=2)
        tempells = reslist[0][1]

        return cls, tempells

    def __call__(self, realization):
        config={'plots_dir': None,
          'min_snr': 10., 'depth_cut': 24.5,
          'mapping': {'wcs': None, 'res': 0.01666666666667,
                      'res_bo': 0.003, 'pad': 0.2,
                      'projection': 'CAR'},
          'band': 'i', 'depth_method': 'fluxerr',
          'shearrot': 'noflip', 'mask_type': 'sirius',
          'ra':  'i_ra', 'dec':  'i_dec',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shape_noise': True,
          'rm_gama09h_region': True,
          'masked_fraction': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/XMM_ceci/masked_fraction.fits',
          'star_catalog': '/projects/HSC/weaklens/s19a_shape_catalog/star_catalog_higher_moments/XMM_psf.csv'}
        logger.info('Running realization : {}.'.format(realization))
        # np.random.seed(realization)
        band = config['band']
        self.mpp = config['mapping']
        fsk, _ = read_flat_map(config['masked_fraction'])
        star_cat = Table.read(config['star_catalog'])

        if 'GAMA09H' in config['star_catalog'] and config['rm_gama09h_region']==True:
            good_seeing_mask = (star_cat[config['ra']]>=132.5)&(star_cat[config['ra']]<=140.)&(star_cat_old[config['dec']]>1.6)    
            logger.info("Good seeing removal %f", (np.sum(good_seeing_mask)/len(star_cat)))
            star_cat.remove_rows(good_seeing_mask)
        # Roohi: move VVDS RAs to be on same side of 0 degrees
        if 'VVDS' in config['star_catalog']:
            logger.info("Shifting star catalog RA by -30 degrees for VVDS")
            change_in_ra = -30.0
            init_ra_vals = star_cat[self.config['ra']].copy()
            star_cat[config['ra']] = init_ra_vals+(np.ones(len(init_ra_vals))*change_in_ra)
            star_cat[config['ra']][star_cat[config['ra']]<0] += 360.0

        star_cat_randomized = self.randomize_shear_cat(star_cat)

        logger.info('Creating e_PSF and T_PSF maps.')
        ePSFstar = self.make_PSF_maps(star_cat_randomized, fsk)

        logger.info('Creating e_PSF and T_PSF residual maps.')
        ePSFresstar = self.make_PSF_res_maps(star_cat_randomized, fsk)

        logger.info('Creating M4_PSF maps.')
        m4PSFstar = self.make_PSF_fourth_moment_maps(star_cat_randomized, fsk)

        logger.info('Creating M4_PSF residual maps.')
        m4PSFresstar = self.make_PSF_res_fourth_moment_maps(star_cat_randomized, fsk)

        f_res = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                                    ePSFresstar[1][0].reshape([fsk.ny,fsk.nx]),
                                    [ePSFresstar[0][0].reshape([fsk.ny,fsk.nx]), ePSFresstar[0][1].reshape([fsk.ny,fsk.nx])],
                                    templates=None)

        
        f_psf = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                                    ePSFstar[1][0].reshape([fsk.ny,fsk.nx]),
                                    [ePSFstar[0][0].reshape([fsk.ny,fsk.nx]), ePSFstar[0][1].reshape([fsk.ny,fsk.nx])],
                                    templates=None)

        
        f_m4_res = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                                    m4PSFresstar[1][0].reshape([fsk.ny,fsk.nx]),
                                    [m4PSFresstar[0][0].reshape([fsk.ny,fsk.nx]), m4PSFresstar[0][1].reshape([fsk.ny,fsk.nx])],
                                    templates=None)


        f_m4_psf = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                                    m4PSFstar[1][0].reshape([fsk.ny,fsk.nx]),
                                    [m4PSFstar[0][0].reshape([fsk.ny,fsk.nx]), m4PSFstar[0][1].reshape([fsk.ny,fsk.nx])],
                                    templates=None)
        #PowerSpecter
        cls = np.zeros((4, 1, self.params['nell']))

        b = nmt.NmtBinFlat(self.params['l0_bins'], self.params['lf_bins'])
        ells_uncoupled = b.get_effective_ells()

        for map_i in range(4):
            j=0
            if map_i == 0:
                probe1 = 'residual'
            elif map_i == 1:
                probe1 = 'leakage'
            elif map_i == 2:
                probe1 = 'm4residual'
            elif map_i == 3:
                probe1 = 'm4leakage'
            spin1 = 2
            spin2 = 2

            logger.info('Computing the power spectrum between probe1 = {} and probe2 = {}.'.format(probe1, probe1))
            logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

            # Define flat sky spin-2 field
            # emaps = [gammamaps[j][0][0].reshape([fsk.ny, fsk.nx]), gammamaps[j][0][1].reshape([fsk.ny, fsk.nx])]
            # f2_1 = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly), self.masks[j],
            #                         emaps, purify_b=False)
            if map_i==0:
                f2_1 = f_res
                f2_2 = f_res
            elif map_i==1:
                f2_1 = f_psf
                f2_2 = f_psf
            elif map_i==2:
                f2_1 = f_m4_res
                f2_2 = f_m4_res
            elif map_i==3:
                f2_1 = f_m4_psf
                f2_2 = f_m4_psf
            # Define flat sky spin-2 field

            if self.wsps[map_i][j] is None:
                logger.info('Workspace element for probe1, probe2 = {}, {} not set.'.format(probe1, probe1))
                logger.info('Computing workspace element.')
                wsp = nmt.NmtWorkspaceFlat()
                wsp.compute_coupling_matrix(f2_1, f2_2, b)
                self.wsps[map_i][j] = wsp
            else:
                logger.info('Workspace element already set for probe1, probe2 = {}, {} not set.'.format(probe1, probe1))
            # Compute pseudo-Cls
            cl_coupled = nmt.compute_coupled_cell_flat(f2_1, f2_2, b)
            # Uncoupling pseudo-Cls
            cl_uncoupled = self.wsps[map_i][j].decouple_cell(cl_coupled)

            # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
            tempclse = cl_uncoupled[0]

            cls[map_i, j, :] = tempclse

        return cls, ells_uncoupled


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