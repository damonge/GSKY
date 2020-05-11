from ceci import PipelineStage
import logging
import numpy as np
import os
import scipy.optimize
import pyccl as ccl
import sacc
from theory.gsky_like import GSKYLike
from theory.predict_theory import GSKYPrediction
import gsky.sacc_utils as sutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HMPARAMS_KEYS = ['HODmod', 'mmin', 'mminp', 'm0', 'm0p', 'm1', 'm1p', 'bhydro', 'massdef', 'pprof']
DEFAULT_COSMO_KEYS = ['Omega_b', 'Omega_k', 'A_s', 'h', 'n_s', 'Omega_c', 'w0', 'wa', 'sigma8']

class LikeMinimizer(PipelineStage) :
    name="LikeMinimizer"
    inputs=[]
    outputs=[]
    config_options={'saccdirs': [str], 'output_run_dir': 'NONE', 'output_dir': 'NONE', 'noisesaccs': 'NONE',
                    'tracers': [str]}

    def get_output_fname(self,name,ext=None):
        fname=self.output_dir+name
        if ext is not None:
            fname+='.'+ext
        return fname

    def parse_input(self):
        """
        Check sanity of input parameters.
        """
        # This is a hack to get the path of the root output directory.
        # It should be easy to get this from ceci, but I don't know how to.
        self.output_dir = self.config['output_dir']+'/'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.ell_max_dict = dict(zip(self.config['tracers'], self.config['ell_max_trc']))

        self.param_keys = self.config['param_keys']

        default_params = self.config['default_params']
        self.cosmo_defaults = {}
        for key in DEFAULT_COSMO_KEYS:
            if key in default_params:
                logger.info('Setting {} to default value {}.'.format(key, default_params[key]))
                self.cosmo_defaults[key] = default_params[key]
        self.hmparams_defaults = {}
        for key in DEFAULT_HMPARAMS_KEYS:
            if key in default_params:
                logger.info('Setting {} to default value {}.'.format(key, default_params[key]))
                self.hmparams_defaults[key] = default_params[key]

        return

    def like_func(self, params):

        param_dict = {'hmparams': self.hmparams_defaults,
                      'cosmo': self.cosmo_defaults}

        for i, key in enumerate(self.param_keys):
            if key in DEFAULT_COSMO_KEYS:
                param_dict['cosmo'][key] = params[i]
            elif key in DEFAULT_HMPARAMS_KEYS:
                param_dict['hmparams'][key] = params[i]
            else:
                raise RuntimeError('Parameter {} not recognized. Aborting.'.format(key))

        if self.bounds is not None:
            if not self.bounds_ok(params):
                logger.info('Parameter out of bounds.')
                like = 1e6
                return like

        try:
            obs_theory = self.gskypred.get_prediction(param_dict)
            like = self.like.computeLike(obs_theory)
            like *= (-1.)
        except BaseException as e:
            logger.error('{} for parameter set {}.'.format(e, params))
            like = 1e6

        return like

    def bounds_ok(self, params):

        if np.any(params < self.bounds_min) or np.any(params > self.bounds_max):
            return False

        else:
            return True

    def minimize(self, minimizer_params):

        if minimizer_params['bounds'] == 'NONE':
            logger.info('No parameter bounds provided.')
            self.bounds = None
        else:
            logger.info('Parameter bounds provided.')
            self.bounds = np.array(minimizer_params['bounds'])
            self.bounds_min = self.bounds[:, 0]
            self.bounds_max = self.bounds[:, 1]

        res = scipy.optimize.minimize(self.like_func, np.array(minimizer_params['x0']), method=minimizer_params['method'],
                                      options={'disp': True, 'ftol': float(minimizer_params['ftol']), 'maxiter':minimizer_params['maxiter']})

        if res.success:
            logger.info('{}'.format(res.message))
            logger.info('Minimizer found minimum at {}.'.format(res.x))
        else:
            logger.info('No minimum found.')

        return res.x

    def run(self):
        """
        Main routine. This stage:
        - Creates gamma1, gamma2 maps and corresponding masks from the reduced catalog for a set of redshift bins.
        - Stores the above into a single FITS file.
        """

        self.parse_input()

        saccfiles = []
        for saccdir in self.config['saccdirs']:
            if self.config['output_run_dir'] != 'NONE':
                path2sacc = os.path.join(saccdir, self.config['output_run_dir']+'/'+'power_spectra_wodpj')
            sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
            logger.info('Read {}.'.format(self.get_output_fname(path2sacc, 'sacc')))
            assert sacc_curr.covariance is not None, 'saccfile {} does not contain covariance matrix. Aborting.'.format(self.get_output_fname(path2sacc, 'sacc'))
            saccfiles.append(sacc_curr)

        if self.config['noisesacc_filename'] != 'NONE':
            logger.info('Reading provided noise saccfile.')
            noise_saccfiles = []
            for i, saccdir in enumerate(self.config['saccdirs']):
                if self.config['output_run_dir'] != 'NONE':
                    path2sacc = os.path.join(saccdir, self.config['output_run_dir'] + '/' + self.config['noisesacc_filename'])
                noise_sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
                logger.info('Read {}.'.format(self.get_output_fname(path2sacc, 'sacc')))
                if noise_sacc_curr.covariance is None:
                    logger.info('noise sacc has no covariance. Adding covariance matrix to noise sacc.')
                    noise_sacc_curr.add_covariance(saccfiles[i].covariance.covmat)
                noise_saccfiles.append(noise_sacc_curr)
            noise_saccfile_coadd = sutils.coadd_saccs(noise_saccfiles, self.config['tracers'], self.ell_max_dict)
        else:
            logger.info('No noise saccfile provided.')
            noise_saccfile_coadd = None
            noise_saccfiles = None

        # Need to coadd saccfiles after adding covariance to noise saccfiles
        saccfile_coadd = sutils.coadd_saccs(saccfiles, self.config['tracers'], self.ell_max_dict)

        if 'theory' in self.config.keys():
            if 'cosmo' in self.config['theory'].keys():
                cosmo_params = self.config['theory']['cosmo']
                cosmo = ccl.Cosmology(**cosmo_params)
            else:
                cosmo = None
            if 'hmparams' in self.config['theory'].keys():
                hmparams = self.config['theory']['hmparams']
            else:
                hmparams = None
        else:
            cosmo = hmparams = None
        self.gskypred = GSKYPrediction(saccfile_coadd, self.config['ells'], hmparams=hmparams, cosmo=cosmo)
        self.like = GSKYLike(saccfile_coadd, noise_saccfile_coadd)

        minimizer_params = self.config['minimizer']
        max_like = self.minimize(minimizer_params)

if __name__ == '__main__':
    cls = PipelineStage.main()

