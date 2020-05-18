import numpy as np
import logging
import os
import copy
import sacc
from theory.predict_theory import GSKYPrediction
import gsky.sacc_utils as sutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_fname(config, name, ext=None):
    fname = config['output_dir'] + '/' + name
    if ext is not None:
        fname += '.' + ext
    return fname

def guess_spectra(params, config):

    if 'ell_max_trc' in config.keys():
        ell_max_dict = dict(zip(config['tracers'], config['ell_max_trc']))
    else:
        ell_max_dict = None

    saccfiles = []
    for saccdir in config['saccdirs']:
        if config['output_run_dir'] != 'NONE':
            path2sacc = os.path.join(saccdir, config['output_run_dir'] + '/' + 'power_spectra_wodpj')
        sacc_curr = sacc.Sacc.load_fits(get_output_fname(config, path2sacc, 'sacc'))
        logger.info('Read {}.'.format(get_output_fname(config, path2sacc, 'sacc')))
        assert sacc_curr.covariance is not None, 'saccfile {} does not contain covariance matrix. Aborting.'.format(
            get_output_fname(config, path2sacc, 'sacc'))
        saccfiles.append(sacc_curr)

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
    noise_saccfile_coadd = sutils.coadd_saccs(noise_saccfiles, config['tracers'], ell_max_dict)

    # Need to coadd saccfiles after adding covariance to noise saccfiles
    saccfile_coadd = sutils.coadd_saccs(saccfiles, config['tracers'], ell_max_dict)

    theor = GSKYPrediction(noise_saccfile_coadd)

    cl_theor = theor.get_prediction(params)

    sacc_guess_spec = copy.deepcopy(noise_saccfile_coadd)
    sacc_guess_spec.mean = noise_saccfile_coadd.mean + cl_theor

    if config['output_run_dir'] != 'NONE':
        input_dir = os.path.join('inputs', config['output_run_dir'])
        input_dir = get_output_fname(config, input_dir)
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
        logger.info(('Created {}.'.format(input_dir)))

    if config['output_run_dir'] != 'NONE':
        coadd_dir = os.path.join('coadds', config['output_run_dir'])
        coadd_dir = get_output_fname(config, coadd_dir)
    if not os.path.isdir(coadd_dir):
        os.makedirs(coadd_dir)
        logger.info(('Created {}.'.format(coadd_dir)))

    saccfile_coadd.save_fits(os.path.join(coadd_dir, 'saccfile_coadd.sacc'), overwrite=True)
    noise_saccfile_coadd.save_fits(os.path.join(coadd_dir, 'noise_saccfile_coadd.sacc'), overwrite=True)
    sacc_guess_spec.save_fits(os.path.join(input_dir, 'saccfile_guess_spectra.sacc'), overwrite=True)
