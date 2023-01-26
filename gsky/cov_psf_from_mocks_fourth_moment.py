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

class CovPSFFromMocksFourthMoment(object):
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

    def add_mbias(self, datIn, mbias, msel, corr, config):
        """
        Rescale the shear by (1 + mbias) following section 5.6 and calculate the
        mock ellipticities according to eq. (24) and (25) of
        https://arxiv.org/pdf/1901.09488.pdf
        Args:
            datIn (ndaray): Original HSC S19A mock catalog (it should haave m=0)
            mbias (float):  The multiplicative bias
            msel (float):   Selection bias [default=0.]
            corr (float):   Correction term for shell thickness, finite resolution and missmatch
                            between n(z_data) and n(z_mock) due to a limited number source planes
        Returns:
            out (ndarray):  Updated S19A mock catalog (with m=mbias)
        """


        # if not isinstance(mbias,(float,int)):
        #     raise TypeError('multiplicative shear estimation bias should be a float.')
        # if not isinstance(msel,(float,int)):
        #     raise TypeError('multiplicative selection bias should be a float.')
        bratio_arr = np.ones(self.nbins+1)
        if 'ntomo_bins' in config:
            self.bin_indxs = config['ntomo_bins']
        else:
            self.bin_indxs = range(self.nbins)
        for ibin in self.bin_indxs:
            # bratio_arr[ibin] = (1+mbias[ibin])*(1+msel[ibin])
            bratio_arr[ibin] = (1+mbias[ibin])*(1+msel[ibin])*corr[ibin]
        # logger.info('bratios: %f %f %f %f %f' % (bratio_arr[0], bratio_arr[1], bratio_arr[2], bratio_arr[3], bratio_arr[4]))
        out   =  datIn.copy()
        # Rescaled gamma by (1+m) and then calculate the distortion delta
        gamma_sq=(out['shear1_sim']**2.+out['shear2_sim']**2.)*bratio_arr[out['tomo_bin']]**2.
        dis1  =  2.*(1-out['kappa'])*out['shear1_sim']*bratio_arr[out['tomo_bin']]/\
                    ((1-out['kappa'])**2+gamma_sq)
        dis2  =  2.*(1-out['kappa'])*out['shear2_sim']*bratio_arr[out['tomo_bin']]/\
                    ((1-out['kappa'])**2+gamma_sq)
        # Calculate the mock ellitpicities
        de    =  dis1*out['noise1_int']+dis2*out['noise2_int'] # for denominators
        dd    =  dis1**2+dis2**2.

        logger.info('dd>1 number of values: %d' % np.sum([dd > 1.0]))
        dd[dd > 1.0] = 1.0

        # avoid dividing by zero (this term is 0 under the limit dd->0)
        tmp1  =  np.divide(dis1,dd,out=np.zeros_like(dd),where=dd!=0)
        tmp2  =  np.divide(dis2,dd,out=np.zeros_like(dd),where=dd!=0)
        # the nominator for e1
        e1_mock= out['noise1_int']+dis1+tmp2*(1-(1-dd)**0.5)*\
            (dis1*out['noise2_int']-dis2*out['noise1_int'])
        # the nominator for e2
        e2_mock= out['noise2_int']+dis2+tmp1*(1-(1-dd)**0.5)*\
            (dis2*out['noise1_int']-dis1*out['noise2_int'])
        # update e1_mock and e2_mock
        out['e1_mock']=e1_mock/(1.+de)+out['noise1_mea']
        out['e2_mock']=e2_mock/(1.+de)+out['noise2_mea']
        return out

    def get_gamma_maps(self, cat, mbias, msel, fsk, config):
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
            if ibin==0:
                g1I_full_arr = []
                g2I_full_arr = []
                weights_full_arr = []
                ra_full_arr = []
                dec_full_arr = []
            if config['shape_noise'] == False:
                gammamaps, gammamasks = createSpin2Map(subcat[config['ra']],
                                                       subcat[config['dec']],
                                                       subcat['shear1_sim']/(1-subcat['kappa']),
                                                       subcat['shear2_sim']/(1-subcat['kappa']), fsk,
                                                       weights=subcat['weight'],
                                                       shearrot=config['shearrot'])
            else:

                erms=   (subcat['noise1_int']**2.+subcat['noise2_int']**2.)/2.
                eres=   1.-np.sum(subcat['weight']*erms)\
                        /np.sum(subcat['weight'])
                # Note: here we assume addtive bias is zero
                # g1I =   subcat['e1_mock']/2./eres
                # g2I =   subcat['e2_mock']/2./eres
                g1I =   subcat['e1_mock']/2./eres/(1.+mbias[ibin])/(1.+msel[ibin])
                g2I =   subcat['e2_mock']/2./eres/(1.+mbias[ibin])/(1.+msel[ibin])
                g1I_full_arr = np.append(g1I_full_arr, g1I)
                g2I_full_arr = np.append(g2I_full_arr, g2I)
                weights_full_arr = np.append(weights_full_arr, subcat['weight'])
                ra_full_arr = np.append(ra_full_arr, subcat[config['ra']])
                dec_full_arr = np.append(dec_full_arr, subcat[config['dec']])
        gammamaps, gammamasks = createSpin2Map(ra_full_arr,
                                               dec_full_arr,
                                               g1I_full_arr,
                                               g2I_full_arr, fsk,
                                               weights_full_arr,
                                               shearrot=config['shearrot'])
        maps_combined = [gammamaps, gammamasks]
        maps.append(maps_combined)

        return maps

    def get_e2rms(self, cat, mbias, msel, config):
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

                erms=   (subcat['noise1_int']**2.+subcat['noise2_int']**2.)/2.
                eres=   1.-np.sum(subcat['weight']*erms)\
                        /np.sum(subcat['weight'])
                # Note: here we assume addtive bias is zero
                # g1I =   subcat['e1_mock']/2./eres
                # g2I =   subcat['e2_mock']/2./eres
                g1I =   subcat['e1_mock']/2./eres/(1.+mbias[ibin])/(1.+msel[ibin])
                g2I =   subcat['e2_mock']/2./eres/(1.+mbias[ibin])/(1.+msel[ibin])

                e1_2rms = np.average((g1I)**2,
                                     weights=subcat['weight'])
                e2_2rms = np.average((g2I)**2,
                                     weights=subcat['weight'])

            e2rms_combined = np.array([e1_2rms, e2_2rms])
            e2rms_arr.append(e2rms_combined)

        return np.array(e2rms_arr)

    def get_w2e2(self, cat, mbias, msel, fsk, config, return_maps=False):
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

                erms=   (subcat['noise1_int']**2.+subcat['noise2_int']**2.)/2.
                eres=   1.-np.sum(subcat['weight']*erms)\
                        /np.sum(subcat['weight'])
                # Note: here we assume addtive bias is zero
                # g1I =   subcat['e1_mock']/2./eres
                # g2I =   subcat['e2_mock']/2./eres
                g1I =   subcat['e1_mock']/2./eres/(1.+mbias[ibin])/(1.+msel[ibin])
                g2I =   subcat['e2_mock']/2./eres/(1.+mbias[ibin])/(1.+msel[ibin])

                w2e2maps_curr = createW2QU2Map(subcat[config['ra']],
                                                       subcat[config['dec']],
                                                       g1I,
                                                       g2I, fsk,
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
          'ra':  'ra_mock', 'dec':  'dec_mock',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shape_noise': True,
          'rm_gama09h_region': True,
          'mocks_dir': '/projects/HSC/weaklens/xlshare/S19ACatalogs/catalog_mock/fields/WIDE12H/',
          'clean_catalog_data': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/WIDE12H_ceci/catalog2/clean_catalog.fits',
          'mock_correction_factors': '/tigress/rdalal/fourier_space_shear/mocks_correction_factor.npy'}

        n_realizations = len(os.listdir(config['mocks_dir']))
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
          'ra':  'ra_mock', 'dec':  'dec_mock',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shape_noise': True,
          'rm_gama09h_region': True,
          'mocks_dir': '/projects/HSC/weaklens/xlshare/S19ACatalogs/catalog_mock/fields/WIDE12H/',
          'selection_array': '/projects/HSC/weaklens/xlshare/S19ACatalogs/photoz_2pt/fiducial_dnnzbin_w95c027/source_sel_WIDE12H.fits',
          'clean_catalog_data': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/WIDE12H_ceci/catalog2/clean_catalog.fits',
          'mock_correction_factors': '/tigress/rdalal/fourier_space_shear/mocks_correction_factor.npy',
          'psf_residual_map': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/WIDE12H_ceci/catalog2/ePSFres_map_psf_used.fits',
          'psf_leakage_map': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/WIDE12H_ceci/catalog2/ePSF_map_psf_used.fits',
          'm4_psf_residual_map': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/WIDE12H_ceci/catalog2/M4_PSFres_map_psf_used.fits',
          'm4_psf_leakage_map': '/tigress/rdalal/fourier_space_shear/GSKY_outputs/WIDE12H_ceci/catalog2/M4_PSF_map_psf_used.fits'}
        logger.info('Running realization : {}.'.format(realization))
        band = config['band']
        self.mpp = config['mapping']
        #map realization number to mock catalog
        r_num = int(np.floor(realization/13))
        r_num_str = format(r_num, '03d')
        rotmat = int(realization - (r_num*13))
        name = 'mock_nres13_r'+r_num_str+'_rotmat'+str(rotmat)+'_shear_catalog.fits'
        # Read catalog
        # logger.info('reading catalog from {}'.format(config['mocks_dir']+name))
        cat = Table.read(config['mocks_dir']+name)
        #ReduceCat

        #Secondary peak cut, binary star cut
        logger.info("Seconday peak cut, binary star cut")
        source_sel_array = Table.read(config['selection_array'])
        cat=cat[source_sel_array['dnnz_bin']>0]

        # Roohi: remove good seeing region in GAMA09H
        if 'GAMA09H' in config['mocks_dir'] and config['rm_gama09h_region']==True:
            good_seeing_mask = (cat[config['ra']]>=132.5)&(cat[config['ra']]<=140.)&(cat[config['dec']]>1.6)    
            logger.info("Good seeing removal %f", (np.sum(good_seeing_mask)/len(cat)))
            cat.remove_rows(good_seeing_mask)

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
        # Get multiplicative bias from data
        hdul1 = fits.open(config['clean_catalog_data']) 
        mhat_arr = np.zeros(4)
        msel_arr = np.zeros(4)
        for i in range(len(mhat_arr)):
            mhat_arr[i] = hdul1[0].header['MHAT_'+str(i+1)]
            msel_arr[i] = hdul1[0].header['MSEL_'+str(i+1)]
        # Correction factor to account for finite resolution, shell thickness, n(z) differences between data and mocks
        # Need to update this, current values are from Xiangchong
        corr_arr=np.load(config['mock_correction_factors'])
        # logger.info('initial e1 mean: %f', (np.mean(cat['e1_mock'])))
        cat = self.add_mbias(cat, mhat_arr, msel_arr, corr_arr, config)
        # logger.info('e1 mean after mbias: %f', (np.mean(cat['e1_mock'])))
        #ShearMapper
        self.nbins = len(config['pz_bins'])-1
        if 'ntomo_bins' in config:
            self.bin_indxs = config['ntomo_bins']
        else:
            self.bin_indxs = range(self.nbins)
        # logger.info('getting e2rms')
        # e2rms = self.get_e2rms(cat, mhat_arr, msel_arr, config)
        # logger.info('getting w2e2')
        # w2e2 = self.get_w2e2(cat, mhat_arr, msel_arr, fsk, config, return_maps=False)
        logger.info('getting gamma maps')
        gammamaps = self.get_gamma_maps(cat, mhat_arr, msel_arr, fsk, config)

        res_hdulist = fits.open(config['psf_residual_map'])
        fsk_res, resmaps = read_flat_map(None, hdu=[res_hdulist[0], res_hdulist[1]])
        _, masks = read_flat_map(None, hdu=[res_hdulist[2], res_hdulist[3]])
        weight = masks[0]
        f_res = nmt.NmtFieldFlat(np.radians(fsk_res.lx), np.radians(fsk_res.ly),
                                    weight.reshape([fsk_res.ny,fsk_res.nx]),
                                    [resmaps[0].reshape([fsk_res.ny,fsk_res.nx]), resmaps[1].reshape([fsk_res.ny,fsk_res.nx])],
                                    templates=None)

        psf_hdulist = fits.open(config['psf_leakage_map'])
        fsk_psf, psfmaps = read_flat_map(None, hdu=[psf_hdulist[0], psf_hdulist[1]])
        _, masks = read_flat_map(None, hdu=[psf_hdulist[2], psf_hdulist[3]])
        weight = masks[0]
        f_psf = nmt.NmtFieldFlat(np.radians(fsk_psf.lx), np.radians(fsk_psf.ly),
                                    weight.reshape([fsk_psf.ny,fsk_psf.nx]),
                                    [psfmaps[0].reshape([fsk_psf.ny,fsk_psf.nx]), psfmaps[1].reshape([fsk_psf.ny,fsk_psf.nx])],
                                    templates=None)

        m4_res_hdulist = fits.open(config['m4_psf_residual_map'])
        fsk_m4_res, m4_resmaps = read_flat_map(None, hdu=[m4_res_hdulist[0], m4_res_hdulist[1]])
        _, masks = read_flat_map(None, hdu=[m4_res_hdulist[2], m4_res_hdulist[3]])
        weight = masks[0]
        f_m4_res = nmt.NmtFieldFlat(np.radians(fsk_m4_res.lx), np.radians(fsk_m4_res.ly),
                                    weight.reshape([fsk_m4_res.ny,fsk_m4_res.nx]),
                                    [m4_resmaps[0].reshape([fsk_m4_res.ny,fsk_m4_res.nx]), m4_resmaps[1].reshape([fsk_m4_res.ny,fsk_m4_res.nx])],
                                    templates=None)

        m4_psf_hdulist = fits.open(config['m4_psf_leakage_map'])
        fsk_m4_psf, m4_psfmaps = read_flat_map(None, hdu=[m4_psf_hdulist[0], m4_psf_hdulist[1]])
        _, masks = read_flat_map(None, hdu=[m4_psf_hdulist[2], m4_psf_hdulist[3]])
        weight = masks[0]
        f_m4_psf = nmt.NmtFieldFlat(np.radians(fsk_m4_psf.lx), np.radians(fsk_m4_psf.ly),
                                    weight.reshape([fsk_m4_psf.ny,fsk_m4_psf.nx]),
                                    [m4_psfmaps[0].reshape([fsk_m4_psf.ny,fsk_m4_psf.nx]), m4_psfmaps[1].reshape([fsk_m4_psf.ny,fsk_m4_psf.nx])],
                                    templates=None)

        #PowerSpecter
        cls = np.zeros((4, 1, self.params['nell']))

        b = nmt.NmtBinFlat(self.params['l0_bins'], self.params['lf_bins'])
        ells_uncoupled = b.get_effective_ells()

        masks = []
        weightmask=True
        if weightmask ==True:
            logger.info('Using weightmask.')
            mask_temp = gammamaps[0][1][0]
            logger.info(mask_temp)
        else:
            logger.info('Using binary mask.')
            mask_temp = gammamaps[0][1][1]
        mask_temp = np.array(mask_temp).reshape([fsk.ny, fsk.nx])
        masks.append(mask_temp)
        
        self.masks = masks

        emaps = [gammamaps[0][0][0].reshape([fsk.ny, fsk.nx]), gammamaps[0][0][1].reshape([fsk.ny, fsk.nx])]
        f2_2 = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly), self.masks[0], emaps, purify_b=False)
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
            probe2 = 'wl'
            spin1 = 2
            spin2 = 2

            logger.info('Computing the power spectrum between probe1 = {} and probe2 = {}.'.format(probe1, probe2))
            logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

            # Define flat sky spin-2 field
            # emaps = [gammamaps[j][0][0].reshape([fsk.ny, fsk.nx]), gammamaps[j][0][1].reshape([fsk.ny, fsk.nx])]
            # f2_1 = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly), self.masks[j],
            #                         emaps, purify_b=False)
            if map_i==0:
                f2_1 = f_res
            elif map_i==1:
                f2_1 = f_psf
            elif map_i==2:
                f2_1 = f_m4_res
            elif map_i==3:
                f2_1 = f_m4_psf
            # Define flat sky spin-2 field

            if self.wsps[map_i][j] is None:
                logger.info('Workspace element for probe1, probe2 = {}, {} not set.'.format(probe1, probe2))
                logger.info('Computing workspace element.')
                wsp = nmt.NmtWorkspaceFlat()
                wsp.compute_coupling_matrix(f2_1, f2_2, b)
                self.wsps[map_i][j] = wsp
            else:
                logger.info('Workspace element already set for probe1, probe2 = {}, {} not set.'.format(probe1, probe2))
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