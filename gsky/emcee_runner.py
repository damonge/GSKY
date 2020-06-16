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
from mcmc_emcee.InitializeFromChain import InitializeFromChain
from mcmc_emcee.SampleFileUtil import SampleFileUtil
from mcmc_emcee.SampleBallPositionGenerator import SampleBallPositionGenerator
import yaml
import gsky.sacc_utils as sutils

import logging

DROP_TRC_COMBS = [['y_0', 'y_0'], ['kappa_0', 'kappa_0'], ['y_0', 'kappa_0'], ['kappa_0', 'y_0']]


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

if 'ell_max_trc' in config.keys():
    ell_max_dict = dict(zip(config['tracers'], config['ell_max_trc']))
else:
    ell_max_dict = None

tracers = config['tracers']

trc_combs = []
if config['fit_comb'] == 'all':
    logger.info('Fitting auto- and cross-correlations of tracers.')
    i = 0
    for tr_i in tracers:
        for tr_j in tracers[:i + 1]:
            if [tr_i, tr_j] not in DROP_TRC_COMBS:
                # Generate the appropriate list of tracer combinations to plot
                trc_combs.append([tr_j, tr_i])
        i += 1
elif config['fit_comb'] == 'auto':
    logger.info('Fitting auto-correlations of tracers.')
    for tr_i in tracers:
        if [tr_i, tr_i] not in DROP_TRC_COMBS:
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
                if [tr_i, tr_j] not in DROP_TRC_COMBS:
                    # Generate the appropriate list of tracer combinations to plot
                    trc_combs.append([tr_i, tr_j])
        i += 1
elif isinstance(config['fit_comb'], list):
    logger.info('Fitting provided tracer combination list.')
    list_intersec = [config['fit_comb'][i] for i in range(len(config['fit_comb'])) if config['fit_comb'][i] in DROP_TRC_COMBS]
    if list_intersec != []:
        logger.info('Dropping unsupported tracer combinations.')
        trc_combs_trim = [config['fit_comb'][i] for i in range(len(config['fit_comb'])) if config['fit_comb'][i] not in DROP_TRC_COMBS]
    trc_combs = config['fit_comb']
else:
    raise NotImplementedError('Only fit_comb = all, auto and cross supported. Aborting.')

logger.info('Fitting tracer combination = {}.'.format(trc_combs))

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
        noise_sacc_curr = sacc.Sacc.load_fits(get_output_fname(config, path2sacc, 'sacc'))
        logger.info('Read {}.'.format(get_output_fname(config, path2sacc, 'sacc')))
        if noise_sacc_curr.covariance is None:
            logger.info('noise sacc has no covariance. Adding covariance matrix to noise sacc.')
            noise_sacc_curr.add_covariance(saccfiles[i].covariance.covmat)
        noise_saccfiles.append(noise_sacc_curr)
    noise_saccfile_coadd = sutils.coadd_saccs(noise_saccfiles, config['tracers'], ell_max_dict=ell_max_dict,
                                              trc_combs=trc_combs)
else:
    logger.info('No noise saccfile provided.')
    noise_saccfile_coadd = None
    noise_saccfiles = None

# Need to coadd saccfiles after adding covariance to noise saccfiles
saccfile_coadd = sutils.coadd_saccs(saccfiles, config['tracers'], ell_max_dict=ell_max_dict, trc_combs=trc_combs)

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
coremod_config['trc_combs'] = trc_combs

th = GSKYCore(saccfile_coadd, coremod_config)
th.setup()
lik = GSKYLike(saccfile_coadd, noise_saccfile_coadd)
lik.setup()

def inrange(p):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p):
    if inrange(p):
        try:
            cl_theory = th.computeTheory(p)
            lnP = lik.computeLikelihood(cl_theory)
        except:
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