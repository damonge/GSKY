#! /usr/bin/env python

import numpy as np
import os
import argparse
import copy
import sacc
import pyccl as ccl
from cosmoHammer import LikelihoodComputationChain
from mcmc.gsky_core_module import GSKYCore, get_params
from mcmc.gsky_like_mcmc import GSKYLike
from mcmc.InitializeFromChain import InitializeFromChain
import yaml
import gsky.sacc_utils as sutils

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_fname(config, name, ext=None):
    fname = config['output_dir'] + name
    if ext is not None:
        fname += '.' + ext
    return fname

def parse_input(config):
    """
    Check sanity of input parameters.
    """
    # This is a hack to get the path of the root output directory.
    # It should be easy to get this from ceci, but I don't know how to.
    config['output_dir'] += '/'
    if not os.path.isdir(config['output_dir']):
        os.makedirs(config['output_dir'])

parser = argparse.ArgumentParser(description='Calculate HSC clustering cls.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', required=True)

args = parser.parse_args()
# print("Likelihood argument:", args.time_likelihood)

logger.info('Read args = {} from command line.'.format(args))

config = yaml.load(open(args.path2config))
logger.info('Read config from {}.'.format(args.path2config))

parse_input(config)

ch_config_params = config['ch_config_params']

logger.info('Called hsc_driver with saccfiles = {}.'.format(config['saccfiles']))
saccs = [sacc.SACC.loadFromHDF(fn) for fn in config['saccfiles']]
logger.info("Loaded {} sacc files.".format(len(saccs)))

if ch_config_params['use_mpi'] == 0:
    from cosmoHammer import CosmoHammerSampler
else:
    from cosmoHammer import MpiCosmoHammerSampler

saccfiles = []
for saccdir in config['saccdirs']:
    if config['output_run_dir'] != 'NONE':
        path2sacc = os.path.join(saccdir, config['output_run_dir'] + '/' + 'power_spectra_wodpj')
    sacc_curr = sacc.Sacc.load_fits(get_output_fname(config, path2sacc, 'sacc'))
    logger.info('Read {}.'.format(get_output_fname(config, path2sacc, 'sacc')))
    assert sacc_curr.covariance is not None, 'saccfile {} does not contain covariance matrix. Aborting.'.format(
        get_output_fname(config, path2sacc, 'sacc'))
    saccfiles.append(sacc_curr)

if config['noisesacc_filename'] != 'NONE':
    logger.info('Reading provided noise saccfile.')
    noise_saccfiles = []
    for i, saccdir in enumerate(config['saccdirs']):
        if config['output_run_dir'] != 'NONE':
            path2sacc = os.path.join(saccdir, config['output_run_dir'] + '/' + config['noisesacc_filename'])
        noise_sacc_curr = sacc.Sacc.load_fits(get_output_fname(path2sacc, 'sacc'))
        logger.info('Read {}.'.format(get_output_fname(path2sacc, 'sacc')))
        if noise_sacc_curr.covariance is None:
            logger.info('noise sacc has no covariance. Adding covariance matrix to noise sacc.')
            noise_sacc_curr.add_covariance(saccfiles[i].covariance.covmat)
        noise_saccfiles.append(noise_sacc_curr)
    noise_saccfile_coadd = sutils.coadd_saccs(noise_saccfiles)
else:
    logger.info('No noise saccfile provided.')
    noise_saccfile_coadd = None
    noise_saccfiles = None

# Need to coadd saccfiles after adding covariance to noise saccfiles
saccfile_coadd = sutils.coadd_saccs(saccfiles)

fit_params = config['fit_params']
if 'theory' in config.keys():
    if 'cosmo' in config['theory'].keys():
        cosmo_params = config['theory']['cosmo']
        cosmo_fit_params = get_params(fit_params, 'cosmo')
        cosmo_default_params = get_params(config['defaults'], 'cosmo')
        assert cosmo_params.keys() <= list(cosmo_fit_params.keys()) + list(cosmo_default_params.keys()), \
            'Provided cosmology params contain keys not specified in fit_params and constants. Aborting.'
        cosmo = ccl.Cosmology(**cosmo_params)
    else:
        cosmo = None
    if 'hmparams' in config['theory'].keys():
        hmparams = config['theory']['hmparams']
        hmparams_fit = get_params(fit_params, 'hmparams')
        hmparams_default = get_params(config['defaults'], 'hmparams')
        assert hmparams.keys() <= list(hmparams_fit.keys()) + list(hmparams_default.keys()), \
            'Provided HM params contain keys not specified in fit_params and constants. Aborting.'
    else:
        hmparams = None
else:
    cosmo = hmparams = None

param_mapping = {}
nparams = len(fit_params.keys())
params = np.zeros((nparams, 4))
for key in fit_params.keys():
    param_mapping[key] = fit_params[key][0]
    params[fit_params[key][0], :] = fit_params[key][1:]


# Set up CosmoHammer
chain = LikelihoodComputationChain(
    min=params[:, 1],
    max=params[:, 2])

tracers = config['tracers']

trc_combs = []
if config['fit_comb'] == 'all':
    logger.info('Fitting auto- and cross-correlations of tracers.')
    i = 0
    for tr_i in tracers:
        for tr_j in tracers[:i + 1]:
            # Generate the appropriate list of tracer combinations to plot
            trc_combs.append([tr_j, tr_i])
        i += 1
elif config['fit_comb'] == 'auto':
    logger.info('Fitting auto-correlations of tracers.')
    for tr_i in tracers:
        trc_combs.append([tr_i, tr_i])
elif config['fit_comb'] == 'cross':
    tracer_type_list = [tr.split('_')[0] for tr in tracers]
    # Get unique tracers and keep ordering
    unique_trcs = []
    [unique_trcs.append(tr) for tr in tracer_type_list if tr not in unique_trcs]
    ntracers0 = tracer_type_list.count(unique_trcs[0])
    ntracers1 = tracer_type_list.count(unique_trcs[1])
    ntracers = np.array([ntracers0, ntracers1])
    logger.info('Fitting cross-correlations of tracers.')
    i = 0
    for tr_i in tracers[:ntracers0]:
        for tr_j in tracers[ntracers0:]:
            if tr_i.split('_')[0] != tr_j.split('_')[0]:
                # Generate the appropriate list of tracer combinations to plot
                trc_combs.append([tr_i, tr_j])
        i += 1
else:
    raise NotImplementedError('Only plot_comb = all, auto and cross supported. Aborting.')

logger.info('Fitting tracer combination = {}.'.format(trc_combs))

coremod_config = copy.deepcopy(config)
coremod_config['param_mapping'] = param_mapping
coremod_config['hmparams'] = hmparams
coremod_config['cosmo'] = cosmo
coremod_config['trc_combs'] = trc_combs
chain.addCoreModule(GSKYCore(saccfile_coadd, coremod_config))

chain.addLikelihoodModule(GSKYLike(saccfile_coadd, noise_saccfile_coadd))

chain.setup()

path2chain = os.path.join('chains', config['output_run_dir'] + '/' + ch_config_params['chainsPrefix'])

if ch_config_params['use_mpi'] == 0:
    if ch_config_params['rerun'] == 0:
        sampler = CosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=get_output_fname(config, path2chain),
            walkersRatio=ch_config_params['walkersRatio'],
            burninIterations=ch_config_params['burninIterations'],
            sampleIterations=ch_config_params['sampleIterations'])
    else:
        assert ch_config_params[
                   'path2rerunchain'] is not None, 'rerun is {}, but path to rerun chains not set. Aborting.'.format(
            ch_config_params['rerun'])

        path2chain = args.path2rerunchain
        sampler = CosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=get_output_fname(config, path2chain),
            walkersRatio=ch_config_params['walkersRatio'],
            burninIterations=ch_config_params['burninIterations'],
            sampleIterations=ch_config_params['sampleIterations'],
            initPositionGenerator=InitializeFromChain(ch_config_params['path2rerunchain'], fraction=0.8))
else:
    if ch_config_params['rerun'] == 0:
        sampler = MpiCosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=os.path.join(ch_config_params['path2output'], ch_config_params['chainsPrefix']),
            walkersRatio=ch_config_params['walkersRatio'],
            burninIterations=ch_config_params['burninIterations'],
            sampleIterations=ch_config_params['sampleIterations'])
    else:
        assert ch_config_params[
                   'path2rerunchain'] is not None, 'rerun is {}, but path to rerun chains not set. Aborting.'.format(
            ch_config_params['rerun'])
        sampler = MpiCosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=os.path.join(ch_config_params['path2output'], ch_config_params['chainsPrefix']),
            walkersRatio=ch_config_params['walkersRatio'],
            burninIterations=ch_config_params['burninIterations'],
            sampleIterations=ch_config_params['sampleIterations'],
            initPositionGenerator=InitializeFromChain(ch_config_params['path2rerunchain'], fraction=0.8))

    sampler.startSampling()

