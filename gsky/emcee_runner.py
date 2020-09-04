#! /usr/bin/env python

import numpy as np
import os
import argparse
import copy
import sys
import sacc
import pyccl as ccl
import emcee
from mcmc_emcee.gsky_core_module import GSKYCore, get_params
from mcmc_emcee.gsky_like_mcmc import GSKYLike
from mcmc_emcee.gauss_prior_like import GaussLike
from mcmc_emcee.InitializeFromChain import InitializeFromChain
from mcmc_emcee.SampleFileUtil import SampleFileUtil
from mcmc_emcee.SampleBallPositionGenerator import SampleBallPositionGenerator
import yaml
import gsky.sacc_utils as sutils

import logging

DROP_TRC_COMBS = [('y_0', 'y_0'), ('kappa_0', 'kappa_0'), ('y_0', 'kappa_0'), ('kappa_0', 'y_0')]


def get_output_fname(config, name, ext=None):
    fname = config['output_dir'] + '/' + name
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

config = yaml.load(open(args.path2config))
ch_config_params = config['ch_params']

path2chain = os.path.join('chains', config['output_run_dir'] + '/' + ch_config_params['chainsPrefix'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(get_output_fname(config, path2chain + '.log'), "w")
ch = logging.StreamHandler()
fh.setLevel(logging.INFO)
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info('Read args = {} from command line.'.format(args))
logger.info('Read config from {}.'.format(args.path2config))

parse_input(config)

if type(config['saccdirs'][0]) == list:
    logger.info('Running with multiple likelihoods.')
    n_likes = len(config['saccdirs'])
    assert type(config['tracers'][0]) == list, 'Multiple likelihoods requested but tracers is not list. Aborting.'
    assert type(config['ell_max_trc'][0]) == list, 'Multiple likelihoods requested but ell_max_trc is not list. Aborting.'
    assert type(config['ells']) == list, 'Multiple likelihoods requested but ells is not list. Aborting.'
    assert type(config['fit_comb']) == list, 'Multiple likelihoods requested but fit_comb is not list. Aborting.'
    assert type(config['noisesacc_filename']) == list, 'Multiple likelihoods requested but noisesacc_filename is not list. Aborting.'
    # Weights for sacc coaddition
    if 'weights' in config.keys():
        assert type(config['weights']) == list, 'Multiple likelihoods requested but weights is not list. Aborting.'
        logger.info('Using cooadd weights = {}.'.format(config['weights']))
        weights = [np.array(config['weights'][i] for i in range(n_likes))]
    else:
        logger.info('No weights provided.')
        weights = [None for i in range(n_likes)]
    if 'path2NGcov' in config.keys():
        assert type(config['path2NGcov']) == list, 'Multiple likelihoods requested but path2NGcov is not list. Aborting.'
    saccfiles = [[] for i in range(n_likes)]
    trc_combs = [[] for i in range(n_likes)]
else:
    logger.info('Running with one likelihood.')
    n_likes = 1
    config['saccdirs'] = [config['saccdirs']]
    config['tracers'] = [config['tracers']]
    if 'ell_max_trc' in config.keys():
        config['ell_max_trc'] = [config['ell_max_trc']]
    if 'ell_min_trc' in config.keys():
        config['ell_min_trc'] = [config['ell_min_trc']]
    config['ells'] = [config['ells']]
    config['fit_comb'] = [config['fit_comb']]
    config['noisesacc_filename'] = [config['noisesacc_filename']]
    saccfiles = [[]]
    trc_combs = [[]]
    # Weights for sacc coaddition
    if 'weights' in config.keys():
        logger.info('Using cooadd weights = {}.'.format(config['weights']))
        weights = [config['weights']]
    else:
        logger.info('No weights provided.')
        weights = [None for i in range(n_likes)]
    if 'path2NGcov' in config.keys():
        config['path2NGcov'] = [config['path2NGcov']]

if 'ell_max_trc' in config.keys():
    ell_max_dict = [dict(zip(config['tracers'][i], config['ell_max_trc'][i])) for i in range(n_likes)]
else:
    ell_max_dict = [None for i in range(n_likes)]
if 'ell_min_trc' in config.keys():
    ell_min_dict = [dict(zip(config['tracers'][i], config['ell_min_trc'][i])) for i in range(n_likes)]
else:
    ell_min_dict = [None for i in range(n_likes)]

for trcs_i in range(len(config['tracers'])):
    tracers = config['tracers'][trcs_i]

    if config['fit_comb'][trcs_i] == 'all':
        logger.info('Fitting auto- and cross-correlations of tracers.')
        i = 0
        for tr_i in tracers:
            for tr_j in tracers[:i + 1]:
                if (tr_i, tr_j) not in DROP_TRC_COMBS:
                    # Generate the appropriate list of tracer combinations to plot
                    trc_combs[trcs_i].append((tr_i, tr_j))
            i += 1
    elif config['fit_comb'][trcs_i] == 'auto':
        logger.info('Fitting auto-correlations of tracers.')
        for tr_i in tracers:
            if (tr_i, tr_i) not in DROP_TRC_COMBS:
                trc_combs[trcs_i].append((tr_i, tr_i))
    elif config['fit_comb'][trcs_i] == 'cross':
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
                    if (tr_i, tr_j) not in DROP_TRC_COMBS:
                        # Generate the appropriate list of tracer combinations to plot
                        trc_combs[trcs_i].append((tr_i, tr_j))
            i += 1
    elif isinstance(config['fit_comb'][trcs_i], list):
        trc_combs[trcs_i] = [tuple(config['fit_comb'][trcs_i][i]) for i in range(len(config['fit_comb'][trcs_i]))]
        logger.info('Fitting provided tracer combination list.')
        list_intersec = [trc_combs[trcs_i][i] for i in range(len(trc_combs[trcs_i])) if trc_combs[trcs_i][i] in DROP_TRC_COMBS]
        if list_intersec != []:
            logger.info('Dropping unsupported tracer combinations.')
            trc_combs[trcs_i] = [trc_combs[trcs_i][i] for i in range(len(trc_combs[trcs_i])) if trc_combs[trcs_i][i] not in DROP_TRC_COMBS]
    else:
        raise NotImplementedError('Only fit_comb = all, auto and cross supported. Aborting.')

    logger.info('Likelihood no = {}.'.format(trcs_i))
    logger.info('Fitting tracer combination = {}.'.format(trc_combs[trcs_i]))

for i in range(n_likes):
    for saccdir in config['saccdirs'][i]:
        if config['output_run_dir'] != 'NONE':
            path2sacc = os.path.join(saccdir, config['output_run_dir'] + '/' + 'power_spectra_wodpj')
        sacc_curr = sacc.Sacc.load_fits(get_output_fname(config, path2sacc, 'sacc'))
        logger.info('Read {}.'.format(get_output_fname(config, path2sacc, 'sacc')))
        assert sacc_curr.covariance is not None, 'saccfile {} does not contain covariance matrix. Aborting.'.format(
            get_output_fname(config, path2sacc, 'sacc'))
        saccfiles[i].append(sacc_curr)

if type(config['saccdirs']) == list:
    noise_saccfiles = [[] for i in range(n_likes)]
    noise_saccfile_coadd = [None for i in range(n_likes)]
else:
    noise_saccfiles = [[]]
    noise_saccfile_coadd = [None]

for i in range(n_likes):
    if config['noisesacc_filename'] != 'NONE':
        logger.info('Reading provided noise saccfile.')
        for ii, saccdir in enumerate(config['saccdirs'][i]):
            if config['output_run_dir'] != 'NONE':
                path2sacc = os.path.join(saccdir, config['output_run_dir'] + '/' + config['noisesacc_filename'][i])
            noise_sacc_curr = sacc.Sacc.load_fits(get_output_fname(config, path2sacc, 'sacc'))
            logger.info('Read {}.'.format(get_output_fname(config, path2sacc, 'sacc')))
            if noise_sacc_curr.covariance is None:
                logger.info('noise sacc has no covariance. Adding covariance matrix to noise sacc.')
                noise_sacc_curr.add_covariance(saccfiles[i][ii].covariance.covmat)
            noise_saccfiles[i].append(noise_sacc_curr)
        # Coadd noise saccs
        if 'conv_win' in config.keys():
            if config['conv_win']:
                if 'coadd_mode' in config.keys():
                    if config['coadd_mode'] == 'inv':
                        logger.info('Performing inverse-variance sacc coaddition.')
                        noise_saccfile_coadd[i] = sutils.coadd_saccs(noise_saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                                  ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i], trim_sacc=False)
                    else:
                        logger.info('Performing weighted sacc coaddition.')
                        noise_saccfile_coadd[i] = sutils.coadd_saccs_separate(noise_saccfiles[i], config['tracers'][i],
                                                                  ell_max_dict=ell_max_dict[i], weights=weights[i],
                                                                  ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i],
                                                                  trim_sacc=False)
                else:
                    logger.info('Performing inverse-variance sacc coaddition.')
                    noise_saccfile_coadd[i] = sutils.coadd_saccs(noise_saccfiles[i], config['tracers'][i],
                                                                 ell_max_dict=ell_max_dict[i], ell_min_dict=ell_min_dict[i],
                                                                 trc_combs=trc_combs[i], trim_sacc=False)
            else:
                if 'coadd_mode' in config.keys():
                    if config['coadd_mode'] == 'inv':
                        logger.info('Performing inverse-variance sacc coaddition.')
                        noise_saccfile_coadd[i] = sutils.coadd_saccs(noise_saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                                  ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i])
                    else:
                        logger.info('Performing weighted sacc coaddition.')
                        noise_saccfile_coadd[i] = sutils.coadd_saccs_separate(noise_saccfiles[i], config['tracers'][i],
                                                                           ell_max_dict=ell_max_dict[i], weights=weights[i],
                                                                           ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i])
                else:
                    logger.info('Performing inverse-variance sacc coaddition.')
                    noise_saccfile_coadd[i] = sutils.coadd_saccs(noise_saccfiles[i], config['tracers'][i],
                                                                 ell_max_dict=ell_max_dict[i], ell_min_dict=ell_min_dict[i],
                                                                 trc_combs=trc_combs[i], trim_sacc=False)
        else:
            if 'coadd_mode' in config.keys():
                if config['coadd_mode'] == 'inv':
                    logger.info('Performing inverse-variance sacc coaddition.')
                    noise_saccfile_coadd[i] = sutils.coadd_saccs(noise_saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                              ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i])
                else:
                    logger.info('Performing weighted sacc coaddition.')
                    noise_saccfile_coadd[i] = sutils.coadd_saccs_separate(noise_saccfiles[i], config['tracers'][i],
                                                                       ell_max_dict=ell_max_dict[i], weights=weights[i],
                                                                       ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i])
            else:
                logger.info('Performing inverse-variance sacc coaddition.')
                noise_saccfile_coadd[i] = sutils.coadd_saccs(noise_saccfiles[i], config['tracers'][i],
                                                             ell_max_dict=ell_max_dict[i], ell_min_dict=ell_min_dict[i],
                                                             trc_combs=trc_combs[i], trim_sacc=False)
    else:
        logger.info('No noise saccfile provided.')
        noise_saccfiles[i] = None
        noise_saccfile_coadd[i] = None

# Need to coadd saccfiles after adding covariance to noise saccfiles
if 'conv_win' in config.keys():
    if config['conv_win']:
        if 'coadd_mode' in config.keys():
            if config['coadd_mode'] == 'inv':
                logger.info('Performing inverse-variance sacc coaddition.')
                saccfile_coadd = [sutils.coadd_saccs(saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                    ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i],
                                                    trim_sacc=False) for i in range(n_likes)]
            else:
                logger.info('Performing weighted sacc coaddition.')
                saccfile_coadd = [sutils.coadd_saccs_separate(saccfiles[i], config['tracers'][i],
                                                             ell_max_dict=ell_max_dict[i], ell_min_dict=ell_min_dict[i],
                                                             weights=weights[i], trc_combs=trc_combs[i],
                                                             trim_sacc=False) for i in range(n_likes)]
        else:
            logger.info('Performing inverse-variance sacc coaddition.')
            saccfile_coadd = [sutils.coadd_saccs(saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                 ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i],
                                                 trim_sacc=False) for i in range(n_likes)]
    else:
        if 'coadd_mode' in config.keys():
            if config['coadd_mode'] == 'inv':
                logger.info('Performing inverse-variance sacc coaddition.')
                saccfile_coadd = [sutils.coadd_saccs(saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                          ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i])
                                                          for i in range(n_likes)]
            else:
                logger.info('Performing weighted sacc coaddition.')
                saccfile_coadd = [sutils.coadd_saccs_separate(saccfiles[i], config['tracers'][i],
                                                                   ell_max_dict=ell_max_dict[i], ell_min_dict=ell_min_dict[i],
                                                                   weights=weights[i], trc_combs=trc_combs[i])
                                                                   for i in range(n_likes)]
        else:
            logger.info('Performing inverse-variance sacc coaddition.')
            saccfile_coadd = [sutils.coadd_saccs(saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                 ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i],
                                                 trim_sacc=False) for i in range(n_likes)]
else:
    if 'coadd_mode' in config.keys():
        if config['coadd_mode'] == 'inv':
            logger.info('Performing inverse-variance sacc coaddition.')
            saccfile_coadd = [sutils.coadd_saccs(saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                                ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i]) for i in range(n_likes)]
        else:
            logger.info('Performing weighted sacc coaddition.')
            saccfile_coadd = [sutils.coadd_saccs_separate(saccfiles[i], config['tracers'][i],
                                                         ell_max_dict=ell_max_dict[i], ell_min_dict=ell_min_dict[i],
                                                         weights=weights[i], trc_combs=trc_combs[i]) for i in range(n_likes)]
    else:
        logger.info('Performing inverse-variance sacc coaddition.')
        saccfile_coadd = [sutils.coadd_saccs(saccfiles[i], config['tracers'][i], ell_max_dict=ell_max_dict[i],
                                             ell_min_dict=ell_min_dict[i], trc_combs=trc_combs[i],
                                             trim_sacc=False) for i in range(n_likes)]

if 'path2NGcov' in config.keys():
    logger.info('path2NGcov provided. Adding NG covariance.')
    for i in range(n_likes):
        cov_NG = np.load(config['path2NGcov'][i])
        assert cov_NG.shape == saccfile_coadd[i].covariance.covmat.shape, 'Shapes of G and NG covariance not consistent. Aborting.'
        logger.info('Read {}.'.format(config['path2NGcov'][i]))
        saccfile_coadd[i].covariance.covmat += cov_NG

for i in range(n_likes):
    if noise_saccfile_coadd[i] is not None and 'path2NGcov' in config.keys():
        logger.info('path2NGcov provided. Adding NG covariance.')
        cov_NG = np.load(config['path2NGcov'][i])
        assert cov_NG.shape == noise_saccfile_coadd[i].covariance.covmat.shape, 'Shapes of G and NG covariance not consistent. Aborting.'
        logger.info('Read {}.'.format(config['path2NGcov'][i]))
        noise_saccfile_coadd[i].covariance.covmat += cov_NG

# Now update trc_combs with sacc ordering
logger.info('Making tr_combs consistent with saccfile ordering.')
for i in range(n_likes):
    trc_comb_curr = saccfile_coadd[i].get_tracer_combinations()
    for ii, (tr_i, tr_j) in enumerate(trc_combs[i]):
        if (tr_j, tr_i) in trc_comb_curr:
            logger.info('Switching order of {}.'.format((tr_i, tr_j)))
            trc_combs[i][ii] = (tr_j, tr_i)

fit_params = config['fit_params']
if 'theory' in config.keys():
    if 'cosmo' in config['theory'].keys():
        cosmo_params = config['theory']['cosmo']
        cosmo_fit_params = get_params(fit_params, 'cosmo')
        cosmo_default_params = get_params(config['constants'], 'cosmo')
        assert cosmo_params.keys() <= set(list(cosmo_fit_params.keys()) + list(cosmo_default_params.keys())), \
            'Provided cosmology params contain keys not specified in fit_params and constants. Aborting.'
        cosmo = ccl.Cosmology(**cosmo_params)
    else:
        cosmo = None
    if 'hmparams' in config['theory'].keys():
        hmparams = config['theory']['hmparams']
        hmparams_fit = get_params(fit_params, 'hmparams')
        hmparams_default = get_params(config['constants'], 'hmparams')
        assert hmparams.keys() <= set(list(hmparams_fit.keys()) + list(hmparams_default.keys())), \
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

coremod_config = copy.deepcopy(config)
coremod_config['param_mapping'] = param_mapping
coremod_config['hmparams'] = hmparams
coremod_config['cosmo'] = cosmo

th = [None for i in range(n_likes)]
lik = [None for i in range(n_likes)]
for i in range(n_likes):
    coremod_config_curr = copy.deepcopy(coremod_config)
    coremod_config_curr['trc_combs'] = trc_combs[i]
    coremod_config_curr['ells'] = coremod_config['ells'][i]
    th[i] = GSKYCore(saccfile_coadd[i], coremod_config_curr)
    th[i].setup()
    lik[i] = GSKYLike(saccfile_coadd[i], noise_saccfile_coadd[i])
    lik[i].setup()

gauss_prior = False
if 'gauss_prior' in config.keys():
    gauss_prior = True
    logger.info('Setting up Gauss prior.')
    assert 'mean' in config['gauss_prior'].keys(), 'Gauss prior requested but no mean supplied. Aborting.'
    assert 'priorcov_filename' in config['gauss_prior'].keys(), 'Gauss prior requested but no priorcov_filename supplied. Aborting.'
    path2cov = os.path.join('inputs', config['output_run_dir'] + '/' + config['gauss_prior']['priorcov_filename'])
    path2cov = get_output_fname(config, path2cov + '.npy')
    cov = np.load(path2cov)

    paramIndx = None
    if 'paramIndx' in config['gauss_prior'].keys():
        if config['gauss_prior']['paramIndx'] != 'NONE':
            paramIndx = config['gauss_prior']['paramIndx']
    else:
        assert config['gauss_prior']['mean'].shape[0] == nparams, 'No paramIndx supplied. Aborting.'

    gauss_prior_lik = GaussLike(config['gauss_prior']['mean'], cov, paramIndx=paramIndx)

def inrange(p):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p):
    if inrange(p):
        lnP = 0.
        try:
            for i in range(n_likes):
                cl_theory = th[i].computeTheory(p)
                lnP += lik[i].computeLikelihood(cl_theory)
            if gauss_prior:
                lnP += gauss_prior_lik.computeLikelihood(p)
        except BaseException as e:
            logger.error('{} for parameter set {}.'.format(e, p))
            lnP = -np.inf
    else:
        lnP = -np.inf
    return lnP

chaindir = os.path.join('chains', config['output_run_dir'])
if not os.path.isdir(get_output_fname(config, chaindir)):
    os.makedirs(get_output_fname(config, chaindir))

nwalkers = nparams * ch_config_params['walkersRatio']

nsteps = ch_config_params['burninIterations'] + ch_config_params['sampleIterations']

if ch_config_params['rerun']:
    assert ch_config_params[
               'path2rerunchain'] is not None, 'rerun is {}, but path to rerun chains not set. Aborting.'.format(
        ch_config_params['rerun'])

    chain_initializer = InitializeFromChain(ch_config_params['path2rerunchain'], fraction=0.8)
    chain_initializer.setup(nparams, nwalkers)
    p_initial = chain_initializer.generate()
else:
    chain_initializer = SampleBallPositionGenerator()
    chain_initializer.setup(params, nwalkers)
    p_initial = chain_initializer.generate()

if ch_config_params['use_mpi'] == 0:

    class DummyPool(object):
        def __init__(self):
            pass

        def is_master(self):
            return True

        def close(self):
            pass

    logger.info('Not using MPI.')
    pool = DummyPool()
    pool_use = None
else:
    logger.info('Using MPI.')
    from schwimmbad import MPIPool

    pool = MPIPool()
    pool_use = pool

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

storageUtil = SampleFileUtil(get_output_fname(config, path2chain))

sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, pool=pool_use)

counter = 1
for pos, prob, _ in sampler.sample(p_initial, iterations=nsteps):
    if pool.is_master():
        logger.info('Iteration done. Persisting.', logging.DEBUG)
        storageUtil.persistSamplingValues(pos, prob)

        if (counter % 10 == 0):
            logger.info('Iteration finished: {}.'.format(counter))

    counter = counter + 1

# Permissions on NERSC
os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')