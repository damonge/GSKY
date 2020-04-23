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

        return

    def coadd_saccs(self, saccfiles, is_noisesacc=False):

        logger.info('Coadding saccfiles.')

        logger.info('Removing B-modes.')
        for saccfile in saccfiles:
            logger.info('Initial size of saccfile = {}.'.format(saccfile.mean.size))
            saccfile.remove_selection(data_type='cl_eb')
            saccfile.remove_selection(data_type='cl_be')
            saccfile.remove_selection(data_type='cl_bb')
            saccfile.remove_selection(data_type='cl_0b')
            logger.info('Size of saccfile after removing B-modes = {}.'.format(saccfile.mean.size))

        for i, saccfile in enumerate(saccfiles):
            sacc_tracers = saccfile.tracers.keys()
            if set(sacc_tracers) == set(self.config['tracers']):
                tempsacc = saccfile
                ind_tmp = i
                logger.info('Found sacc with all requested tracers at {}.'.format(ind_tmp))
                break

        try:
            coadd_mean = tempsacc.mean
            if not is_noisesacc:
                coadd_cov = tempsacc.covariance.covmat
            datatypes = tempsacc.get_data_types()

            nmeans = np.ones_like(coadd_mean)
            if not is_noisesacc:
                ncovs = np.ones_like(coadd_cov)

        except:
            raise RuntimeError('More tracers requested than contained in any of the provided sacc files. Aborting.')

        for i, saccfile in enumerate(saccfiles):
            if i != ind_tmp:
                sacc_tracers = saccfile.tracers.keys()
                if set(sacc_tracers).issubset(self.config['tracers']) and len(sacc_tracers) < len(self.config['tracers']):
                    missing_tracers = list(set(self.config['tracers']) - set(sacc_tracers))
                    logger.info('Found missing tracers {} in saccfile {}.'.format(missing_tracers, i))

                for datatype in datatypes:
                    tracer_combs = tempsacc.get_tracer_combinations(data_type=datatype)
                    for tr_i, tr_j in tracer_combs:
                        if not is_noisesacc:
                            _, cl, cov = saccfile.get_ell_cl(datatype, tr_i, tr_j, return_cov=True)
                        else:
                            _, cl = saccfile.get_ell_cl(datatype, tr_i, tr_j, return_cov=False)

                        ind_here = saccfile.indices(data_type=datatype, tracers=(tr_i, tr_j))
                        ind_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i, tr_j))
                        if not ind_here.size == 0:
                            coadd_mean[ind_tempsacc] += cl
                            nmeans[ind_tempsacc] += 1
                            if not is_noisesacc:
                                coadd_cov[ind_tempsacc][:, ind_tempsacc] += cov
                                ncovs[ind_tempsacc][:, ind_tempsacc] += 1

        coadd_mean /= nmeans
        if not is_noisesacc:
            coadd_cov /= ncovs ** 2

        # Copy sacc
        saccfile_coadd = tempsacc.copy()
        # Set mean of new saccfile to coadded mean
        saccfile_coadd.mean = coadd_mean
        if not is_noisesacc:
            saccfile_coadd.add_covariance(coadd_cov)

        return saccfile_coadd

    def like_func(self, params):

        try:
            obs_theory = self.gskypred.get_prediction(params)
            like = self.like.computeLike(obs_theory)
            like *= (-1.)
        except BaseException as e:
            logger.error('{} for parameter set {}.'.format(e, params))
            like = 1e10

        return like

    def minimize(self, minimizer_params):

        res = scipy.optimize.minimize(self.like_func, np.array(minimizer_params['x0']), method=minimizer_params['method'], bounds=minimizer_params['bounds'],
                                options={'disp': True, 'ftol': int(minimizer_params['ftol']), 'maxiter':minimizer_params['maxiter']})

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
        saccfile_coadd = self.coadd_saccs(saccfiles)

        if self.config['noisesacc_filename'] != 'NONE':
            logger.info('Reading provided noise saccfile.')
            noise_saccfiles = []
            for saccdir in self.config['saccdirs']:
                if self.config['output_run_dir'] != 'NONE':
                    path2sacc = os.path.join(saccdir, self.config['output_run_dir'] + '/' + self.config['noisesacc_filename'])
                noise_sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
                logger.info('Read {}.'.format(self.get_output_fname(path2sacc, 'sacc')))
                noise_saccfiles.append(noise_sacc_curr)
            noise_saccfile_coadd = self.coadd_saccs(noise_saccfiles, is_noisesacc=True)
        else:
            logger.info('No noise saccfile provided.')
            noise_saccfile_coadd = None
            noise_saccfiles = None

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

