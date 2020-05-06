from ceci import PipelineStage
import logging
import numpy as np
import os
import scipy.optimize
import pyccl as ccl
import sacc
from theory.gsky_like import GSKYLike
from theory.predict_theory import GSKYPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HMPARAMS_KEYS = ['HODmod', 'mmin', 'mminp', 'm0', 'm0p', 'm1', 'm1p', 'bhydro', 'massdef', 'pprof']
DEFAULT_COSMO_KEYS = ['Omega_b', 'Omega_k', 'A_s', 'h', 'n_s', 'Omega_c', 'w0', 'wa']

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
                logger.info('Setting {} to default value {}.'.format((key, default_params[key])))
                self.cosmo_defaults[key] = default_params[key]
        self.hmparams_defaults = {}
        for key in DEFAULT_HMPARAMS_KEYS:
            if key in default_params:
                logger.info('Setting {} to default value {}.'.format((key, default_params[key])))
                self.hmparams_defaults[key] = default_params[key]

        return

    def coadd_saccs(self, saccfiles):

        logger.info('Coadding all saccfiles weighted by inverse variance.')

        for saccfile in saccfiles:
            logger.info('Initial size of saccfile = {}.'.format(saccfile.mean.size))
            logger.info('Removing B-modes.')
            saccfile.remove_selection(data_type='cl_eb')
            saccfile.remove_selection(data_type='cl_be')
            saccfile.remove_selection(data_type='cl_bb')
            saccfile.remove_selection(data_type='cl_0b')
            logger.info('Removing yxy.')
            saccfile.remove_selection(data_type='cl_00', tracers=('y_0', 'y_0'))
            logger.info('Removing kappaxkappa.')
            saccfile.remove_selection(data_type='cl_00', tracers=('kappa_0', 'kappa_0'))
            logger.info('Removing kappaxy.')
            saccfile.remove_selection(data_type='cl_00', tracers=('kappa_0', 'y_0'))
            saccfile.remove_selection(data_type='cl_00', tracers=('y_0', 'kappa_0'))
            logger.info('Size of saccfile after cuts = {}.'.format(saccfile.mean.size))

            logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
            for tr_i, tr_j in saccfile.get_tracer_combinations():
                ell_max_curr = min(self.ell_max_dict[tr_i], self.ell_max_dict[tr_j])
                logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
                saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
            logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

        ntracers_arr = np.array([len(saccfile.tracers) for saccfile in saccfiles])
        ntracers_unique = np.unique(ntracers_arr)[::-1]

        saccs_list = [[] for i in range(ntracers_unique.shape[0])]
        for i in range(ntracers_unique.shape[0]):
            for saccfile in saccfiles:
                if len(saccfile.tracers) == ntracers_unique[i]:
                    saccs_list[i].append(saccfile)

        sacc_coadds = [0 for i in range(ntracers_unique.shape[0])]
        for i in range(ntracers_unique.shape[0]):
            len_curr = ntracers_unique[i]
            nsacc_curr = len(saccs_list[i])
            logger.info('Found {} saccfiles of length {}.'.format(nsacc_curr, len_curr))
            for j, saccfile in enumerate(saccs_list[i]):
                if j == 0:
                    coadd_mean = saccfile.mean
                    coadd_cov = saccfile.covariance.covmat
                else:
                    coadd_mean += saccfile.mean
                    coadd_cov += saccfile.covariance.covmat

            coadd_mean /= nsacc_curr
            coadd_cov /= nsacc_curr ** 2

            # Copy sacc
            saccfile_coadd = saccfile.copy()
            # Set mean of new saccfile to coadded mean
            saccfile_coadd.mean = coadd_mean
            saccfile_coadd.add_covariance(coadd_cov)
            sacc_coadds[i] = saccfile_coadd

        tempsacc = sacc_coadds[0]
        tempsacc_tracers = tempsacc.tracers.keys()
        datatypes = tempsacc.get_data_types()
        invcov_coadd = np.linalg.inv(tempsacc.covariance.covmat)
        mean_coadd = np.dot(invcov_coadd, tempsacc.mean)

        assert set(tempsacc_tracers) == set(self.config['tracers']), 'Different tracers requested than present in largest ' \
                                                                     'saccfile. Aborting.'

        for i, saccfile in enumerate(sacc_coadds[1:]):
            sacc_tracers = saccfile.tracers.keys()
            missing_tracers = list(set(self.config['tracers']) - set(sacc_tracers))
            logger.info('Found missing tracers {} in saccfile {}.'.format(missing_tracers, i))

            invcov_small_curr = np.linalg.inv(saccfile.covariance.covmat)

            mean_big_curr = np.zeros_like(tempsacc.mean)
            invcov_big_curr = np.zeros_like(tempsacc.covariance.covmat)

            for datatype in datatypes:
                tracer_combs = tempsacc.get_tracer_combinations(data_type=datatype)
                for tr_i1, tr_j1 in tracer_combs:
                    _, cl = saccfile.get_ell_cl(datatype, tr_i1, tr_j1, return_cov=False)

                    ind_here = saccfile.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                    ind_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                    if not ind_here.size == 0:
                        mean_big_curr[ind_tempsacc] = cl
                    for tr_i2, tr_j2 in tracer_combs:
                        ind_i1j1_curr = saccfile.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                        ind_i2j2_curr = saccfile.indices(data_type=datatype, tracers=(tr_i2, tr_j2))

                        subinvcov_curr = invcov_small_curr[np.ix_(ind_i1j1_curr, ind_i2j2_curr)]

                        ind_i1j1_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                        ind_i2j2_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i2, tr_j2))

                        if ind_i1j1_curr.size != 0 and ind_i2j2_curr.size != 0:
                            invcov_big_curr[np.ix_(ind_i1j1_tempsacc, ind_i2j2_tempsacc)] = subinvcov_curr

            mean_coadd += np.dot(invcov_big_curr, mean_big_curr)
            invcov_coadd += invcov_big_curr

        # Copy sacc
        saccfile_coadd = tempsacc.copy()
        # Set mean of new saccfile to coadded mean
        cov_coadd = np.linalg.inv(invcov_coadd)
        saccfile_coadd.mean = np.dot(cov_coadd, mean_coadd)
        saccfile_coadd.add_covariance(cov_coadd)

        return saccfile_coadd

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
            obs_theory = self.gskypred.get_prediction(params)
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
            noise_saccfile_coadd = self.coadd_saccs(noise_saccfiles, is_noisesacc=True)
        else:
            logger.info('No noise saccfile provided.')
            noise_saccfile_coadd = None
            noise_saccfiles = None

        # Need to coadd saccfiles after adding covariance to noise saccfiles
        saccfile_coadd = self.coadd_saccs(saccfiles)

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
        self.gskypred = GSKYPrediction(saccfile_coadd, self.config['ells'], self.config['param_keys'], hmparams=hmparams,
                                       cosmo=cosmo)
        self.like = GSKYLike(saccfile_coadd, noise_saccfile_coadd)

        minimizer_params = self.config['minimizer']
        max_like = self.minimize(minimizer_params)

if __name__ == '__main__':
    cls = PipelineStage.main()

