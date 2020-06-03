#! /usr/bin/env python

import numpy as np
#import healpy as hp
import copy
from astropy.io import fits
import pymaster as nmt
from ..map_utils import createSpin2Map
from ..flatmaps import read_flat_map

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseMaps(object):
    """
    Class to generate noise-only maps to be used when simulating
    multiprobe surveys.
    """

    def __init__(self, noiseparams={}):
        """
        Constructor for the NoiseMaps class
        """

        self.params = noiseparams
        self.setup()
        self.print_params()

    def print_params(self):
        """
        Prints the parameter combination chosen to initialise NoiseMaps.
        """

        logger.info('NoiseMaps has been initialised with the following attributes:')
        for key in self.params.keys():
            print('{} = {}'.format(key, self.params[key]))

    def generate_maps(self, signalmaps=None):
        """
        Generates a list of noise-only maps for the defined probes.
        These are:
        CMB temperature: random permutation of the HMHD map
        galaxy overdensity: overdensity map generated by randomly placing
        Ngal galaxies in the survey footprint
        gamma: galaxy ellipticity map generated by rotating the shear of
         each galaxy by a random angle
        :param shearmap: 2D list with noisefree cosmic shear maps for e1 and e1 to
        add to the noisy ellipticities
        default: None
        :return maps: 1D list with map[0]=cmbnoisemap, map[1]=galaxy noise map,
        map[2]=ellipticity noise map
        """

        logger.info('Generating noise maps.')
        data = self.read_data()
        maps = [0 for i in range(self.params['nmaps'])]
        ii = 0
        for i, tracer in enumerate(self.params['tracers']):
            probe = self.params['probes'][i]
            logger.info('Generating map for tracer = {}.'.format(tracer))
            logger.info('Generating map for probe = {}.'.format(probe))
            if probe != 'galaxy_shear':
                if self.params['noisemodel'][i] == 'data':
                    maps[ii] = self.datanoisemap(probe, data[tracer])
                else:
                    maps[ii] = self.gaussnoisemap(probe, data[tracer])
                ii += 1
            else:
                if self.params['noisemodel'][i] == 'data':
                    if signalmaps is not None:
                        shearmaps = [signalmaps[i], signalmaps[i+self.params['nspin2']]]
                    else:
                        shearmaps = signalmaps

                    tempmaps = self.datanoisemap(probe, data[tracer], shearmaps)
                else:
                    tempmaps = self.gaussnoisemap(probe, data[tracer])
                maps[ii] = tempmaps[0]
                maps[ii+1] = tempmaps[1]
                ii += 2

        if self.params['nspin2'] > 0:
            logger.info('Spin 2 fields present. Reordering maps.')
            reordered_maps = self.reorder_maps(maps)

            if self.params['nspin2'] == 1:
                assert np.sum([np.all(maps[i] == reordered_maps[i]) for i in range(len(maps))]) == len(maps), \
                    'Something went wrong with map reordering.'
        else:
            logger.info('No spin 2 fields. Keeping map ordering.')
            reordered_maps = copy.deepcopy(maps)

        return reordered_maps

    def reorder_maps(self, maps):

        tempmaps = copy.deepcopy(maps)

        spins = np.array(self.params['spins'])
        nspin2 = np.sum(spins == 2)
        ind = np.where(spins == 2)[0]
        min_ind = np.amin(ind)
        tempmaps[min_ind: min_ind+nspin2] = maps[min_ind::2]
        tempmaps[min_ind+nspin2:] = maps[min_ind+1::2]

        return tempmaps

    def datanoisemap(self, probe, data, shearmaps=None):
        """
        Generates a noise-only HEALPix map for the specified probe using real data
        :param probe: string tag of desired probe
        :param data: the data needed to generate the noisemap
        CMB temperature & galaxy overdensity: HEALPix map
        gamma: structured array with galaxy position and ellipticity catalog
        :param shearmap: 2D list with noisefree cosmic shear maps for e1 and e1 to
        add to the noisy ellipticities
        default: None
        :return noisemap: HEALPix map of the noise for the respective probe
        """

        if probe == 'galaxy_density':
            noisemap = self.randomize_deltag_map(data)

        elif probe == 'galaxy_shear':
            # Add the cosmic shear to the noise ellipticites
            if shearmaps is not None:
                logger.info('Adding signal map to ellipticities.')
                flatmap = data['fsk'].pos2pix(data['shearcat']['ra'], data['shearcat']['dec'])

                data['shearcat']['ishape_hsm_regauss_e1_calib'] += shearmaps[0].flatten()[flatmap]
                data['shearcat']['ishape_hsm_regauss_e2_calib'] += shearmaps[1].flatten()[flatmap]

            noisemap = self.randomize_shear_map(data)

        else:
            raise NotImplementedError('Probes other than galaxy_density and galaxy_shear'\
                                      ' not implemented yet. Aborting.')

        return noisemap

    def randomize_shear_map(self, data):

        logger.info('Randomizing shear map.')

        randomized_cat = self.randomize_shear_cat(data['shearcat'])

        gammamaps, gammamasks = createSpin2Map(randomized_cat['ra'], randomized_cat['dec'], randomized_cat['ishape_hsm_regauss_e1_calib'], \
                                               randomized_cat['ishape_hsm_regauss_e2_calib'], data['fsk'], \
                                               weights=randomized_cat['ishape_hsm_regauss_derived_shape_weight'], \
                                               shearrot=self.params['shearrot'])

        return gammamaps[0].reshape([data['fsk'].ny, data['fsk'].nx]), gammamaps[1].reshape([data['fsk'].ny, data['fsk'].nx])

    def randomize_shear_cat(self, cat):
        """
        Rotates each galaxy ellipticity from the galaxy catalog data by a random angle to
        eliminate correlations between galaxy shapes.
        This is used to estimate the shape noise contribution to the shear power spectrum.
        :param cat: structured array with galaxy catalog to randomise
        :return randomiseddata: structured array with galaxy catalog with randomised ellipticities
        """

        logger.info('Randomizing shear catalogue.')

        assert 'ishape_hsm_regauss_e1_calib' in cat.dtype.names, \
        logger.warning('Catalog needs to contain calibrated shear columns.')

        # Copy the input data so it does not get overwritten
        randomisedcat = copy.deepcopy(cat)

        # Seed the random number generator
        np.random.seed(seed=None)

        thetarot = 2.*np.pi*np.random.random_sample((cat['ishape_hsm_regauss_e1_calib'].shape[0], ))

        randomisedcat['ishape_hsm_regauss_e1_calib'] = np.cos(2*thetarot)*cat['ishape_hsm_regauss_e1_calib'] - \
                                                 np.sin(2*thetarot)*cat['ishape_hsm_regauss_e2_calib']

        randomisedcat['ishape_hsm_regauss_e2_calib'] = np.sin(2*thetarot)*cat['ishape_hsm_regauss_e1_calib'] + \
                                                 np.cos(2*thetarot)*cat['ishape_hsm_regauss_e2_calib']

        return randomisedcat

    def gaussnoisemap(self, probe, data):
        """
        Generates a noise-only HEALPix map as a Gaussian realisation of a theoretical noise power spectrum
        :param probe: string tag of desired probe
        :param data: the power spectrum needed to generate the noisemap
        :return noisemap: HEALPix map of the noise for the respective probe
        """

        if probe == 'galaxy_shear':
            # In the new healpy ordering the order of the power spectra is
            # TT, EE, BB, TE, EB, TB
            # and one needs to set at least 4 of those
            # We set the E and B mode power spectra to the theoretical shape noise power spectrum
            zeroarr = np.zeros(data['noisecls'].shape[0])
            cl_inp = [data['noisecls'], zeroarr, data['noisecls']]
            noisemap = nmt.synfast_flat(data['fsk'].nx, data['fsk'].ny, np.radians(data['fsk'].lx),
                                        np.radians(data['fsk'].ly), cl_inp, spin_arr=[2], seed=-1, beam=None)
        else:
            noisemap = nmt.synfast_flat(data['fsk'].nx, data['fsk'].ny, np.radians(data['fsk'].lx),
                                        np.radians(data['fsk'].ly), [data['noisecls']], spin_arr=[0],
                                        seed=-1, beam=None)

        return noisemap

    def read_data(self):
        """
        Reads in the data needed for the generation of noise-only maps and
        saves them as class attributes.
        These are:
        CMB temperature: noise only map (half mission half difference map)
        galaxy overdensity: boolean mask of galaxy survey footprint
        gamma: structured array with galaxy position and ellipticity catalog
        :param:
        :return data: dictionary with data needed to generate noise realisations for each
        probe
        """

        # Read quantities needed by all probes
        assert 'path2fsk' in self.params, 'path2fsk parameter not provided. Aborting.'
        logger.info("Reading masked fraction from {}.".format(self.params['path2fsk']))
        fsk, _ = read_flat_map(self.params['path2fsk'])

        data = {}
        for i, tracer in enumerate(self.params['tracers']):
            probe = self.params['probes'][i]
            logger.info('Reading noise data for tracer = {}.'.format(tracer))
            logger.info('Reading noise data for probe = {}.'.format(probe))
            logger.info('Noisemodel = {}.'.format(self.params['noisemodel'][i]))

            if self.params['noisemodel'][i] == 'data':

                if probe == 'galaxy_shear':
                    # assert 'shearrot' in self.params, 'Requesting noise model from data but shearrot parameter not provided. Aborting.'
                    assert 'path2shearcat' in self.params, 'Requesting noise model from data but path2shearcat parameter not provided. Aborting.'

                    data[tracer] = {}

                    hdulist = fits.open(self.params['path2shearcat'])
                    cat = hdulist[1].data
                    logger.info('Read {}.'.format(self.params['path2shearcat']))

                    # Remove masked objects
                    logger.info('Removing masked objects.')
                    if self.params['mask_type'] == 'arcturus':
                        logger.info('Applying mask_type = {}.'.format(self.params['mask_type']))
                        msk = cat['mask_Arcturus'].astype(bool)
                    elif self.params['mask_type'] == 'sirius':
                        logger.info('Applying mask_type = {}.'.format(self.params['mask_type']))
                        msk = np.logical_not(cat['iflags_pixel_bright_object_center'])
                        msk *= np.logical_not(cat['iflags_pixel_bright_object_any'])
                    else:
                        raise KeyError("Mask type " + self.params['mask_type'] +
                                       " not supported. Choose arcturus or sirius")
                    logger.info('Applying FDFC mask.')
                    msk *= cat['wl_fulldepth_fullcolor']
                    cat = cat[msk]

                    if 'ntomo_bins' in self.params.keys():
                        logger.info('Selecting galaxies falling in tomographic bin {}.'.format(self.params['ntomo_bins'][i]))
                        if 'shear_cat' in cat.dtype.names:
                            logger.info('Applying shear cuts to catalog')
                            logger.info('Initial size = {}.'.format(cat['ra'].shape))
                            if self.params['ntomo_bins'][i] != -1:
                                mask = (cat['tomo_bin']==self.params['ntomo_bins'][i]) & (cat['shear_cat'])
                            else:
                                mask = (cat['tomo_bin'] >= 0) & (cat['shear_cat'])
                            cat = cat[mask]
                            logger.info('Size after cut = {}.'.format(cat['ra'].shape))
                        else:
                            logger.info('Initial size = {}.'.format(cat['ra'].shape))
                            if self.params['ntomo_bins'][i] != -1:
                                mask = (cat['tomo_bin'] == self.params['ntomo_bins'][i])
                            else:
                                mask = (cat['tomo_bin'] >= 0)
                            cat = cat[mask]
                            logger.info('Size after cut = {}.'.format(cat['ra'].shape))
                    else:
                        if 'shear_cat' in cat.dtype.names:
                            logger.info('Applying shear cuts to catalog')
                            logger.info('Initial size = {}.'.format(cat['ra'].shape))
                            mask = (cat['shear_cat'])
                            cat = cat[mask]
                            logger.info('Size after cut = {}.'.format(cat['ra'].shape))

                    data[tracer]['shearcat'] = cat
                    data[tracer]['fsk'] = fsk

                    if self.params['posfromshearcat'] == 0:
                        assert 'path2shearmask' in self.params, 'Requesting randomized galaxy positions for gamma but path2shearmask not provided. Aborting.'
                        tempmap = read_flat_map(self.params['path2shearmask'], i_map=6*i+3)
                        data[tracer]['shearmask'] = tempmap

                elif probe == 'galaxy_density':
                    assert 'Ngal' in self.params, 'Requesting noise model from data but Ngal parameter not provided. Aborting.'
                    assert 'galaxy_density_mask' in self.params, 'Requesting noise model from data but galaxy_density'\
                                                                 '_mask not provided. Aborting.'

                    data[tracer] = {}

                    tempmask = self.params['galaxy_density_mask'].astype('bool').astype('int')
                    data[tracer]['msk'] = tempmask
                    data[tracer]['fsk'] = fsk
                    data[tracer]['msk_bi'] = self.params['galaxy_density_msk_bi']
                    data[tracer]['mskfrac'] = self.params['galaxy_density_mskfrac']
                    data[tracer]['Ngal'] = self.params['Ngal'][self.params['ntomo_bins'][i]]

                else:
                    raise NotImplementedError('Probes other than galaxy_density, galaxy_shear not implemented at the moment. Aborting.')

            else:
                assert 'path2noisecls' in self.params, 'Requesting theretical noise model but path2noisecls parameter not provided. Aborting.'

                data[tracer] = {}

                data[tracer]['noisecls'] = np.genfromtxt(self.params['path2noisecls'][i], usecols={1})
                logger.info('Read {}.'.format(self.params['path2noisecls'][i]))
                data[tracer]['fsk'] = fsk

        return data

    def randomize_deltag_map(self, data):
        """
        Creates a randomised version of the input map map by assigning the
        galaxies in the surevy to random pixels in the map. Basically it rotates each
        galaxy by a random angle but not rotating it out of the survey footprint.
        :param map: masked galaxy overdensity map which needs to randomised
        :param Ngal: number of galaxies used to create the map
        :return randomised_map: a randomised version of the masked input map
        """

        logger.info('Randomizing galaxy map.')

        mask = data['msk']
        Ngal = data['Ngal']

        np.random.seed(seed=None)
        maskpix = np.where(mask == 1.)[0]

        galpix = np.random.choice(np.arange(maskpix.shape[0]), size=Ngal)

        random_nmap = np.bincount(galpix, minlength=mask.shape[0])

        goodpix = np.where(data['msk_bi'] > 0.1)[0]
        ndens = np.sum(random_nmap * data['msk_bi']) / np.sum(mask)
        delta = np.zeros_like(mask)
        delta[goodpix] = random_nmap[goodpix] / (ndens * data['mskfrac'][goodpix]) - 1

        # Reshape map
        delta = delta.reshape([data['fsk'].ny, data['fsk'].nx])

        return delta

    def setup(self):
        """
        Sets up derived parameters from the input parameters.
        :return:
        """

        logger.info('Setting up NoiseMaps module.')

        self.params['nmaps'] = len(self.params['probes']) + np.sum(self.params['spins'] == 2)

        logger.info('Setup done!')










