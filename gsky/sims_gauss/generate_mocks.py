#! /usr/bin/env python

import argparse
import logging
import numpy as np
import yaml
import os
from ..flatmaps import read_flat_map
from .MockSurvey import MockSurvey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Generate a suite of Gaussian mocks.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

### END OF PARSER ###

if __name__ == '__main__':

    args = parser.parse_args()

    config = yaml.load(open(args.path2config))
    logger.info('Read config from {}.'.format(args.path2config))

    nrealiz = config['simparams']['nrealiz']

    logger.info("Reading masked fraction from {}.".format(config['simparams']['path2fsk']))
    mask, _ = read_flat_map(config['simparams']['path2fsk'])

    # Here assuming for simplicity that masks are the same
    masks = [mask, mask, mask, mask, mask, mask]

    if 'l0_bins' in config['simparams']:
        config['simparams']['l0_bins'] = np.array(config['simparams']['l0_bins'])
    if 'l1_bins' in config['simparams']:
        config['simparams']['l1_bins'] = np.array(config['simparams']['l1_bins'])
    if 'spins' in config['simparams']:
        config['simparams']['spins'] = np.array(config['simparams']['spins'])

    if config['noiseparams'] is not None:
        logger.info('Generating noisy mocks.')
        mocksurvey = MockSurvey(masks, config['simparams'], config['noiseparams'])
    else:
        logger.info('Generating noise-free mocks.')
        mocksurvey = MockSurvey(masks, config['simparams'], noiseparams={})

    cls, noisecls, ells, wsps = mocksurvey.reconstruct_cls_parallel()

    if not os.path.isdir(config['path2outputdir']):
        try:
            os.makedirs(config['path2outputdir'])
        except:
            pass

    path2clarr = os.path.join(config['path2outputdir'], 'cls_signal-noise-removed_nrealis={}.npy'.format(nrealiz))
    np.save(path2clarr, cls)
    logger.info('Written signal cls to {}.'.format(path2clarr))

    path2clnoisearr = os.path.join(config['path2outputdir'], 'cls_noise_nrealis={}.npy'.format(nrealiz))
    np.save(path2clnoisearr, noisecls)
    logger.info('Written noise cls to {}.'.format(path2clnoisearr))

    path2ellarr = os.path.join(config['path2outputdir'], 'ells_uncoupled_nrealis={}.npy'.format(nrealiz))
    np.save(path2ellarr, ells)
    logger.info('Written ells to {}.'.format(path2ellarr))

    for i in range(config['simparams']['nprobes']):
        for ii in range(i+1):
            path2wsp = os.path.join(config['path2outputdir'], 'wsp_probe1={}_probe2={}.dat'.format(i, ii))
            wsps[i][ii].write_to(str(path2wsp))
            logger.info('Written wsp for probe1 = {} and probe2 = {} to {}.'.format(i, ii, path2wsp))
