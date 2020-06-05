from ceci import PipelineStage
import numpy as np
import logging
import os
import copy
import scipy.interpolate
from astropy.io import fits
import pymaster as nmt
from .flatmaps import read_flat_map
from .types import FitsFile, DummyFile
import sacc
from theory.predict_theory import GSKYPrediction
import gsky.sacc_utils as sutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuessSpecter(PipelineStage) :
    name="GuessSpecter"
    inputs=[('masked_fraction', FitsFile), ('depth_map', FitsFile),
            ('gamma_maps', FitsFile), ('act_maps', FitsFile)]
    outputs=[('dummy', DummyFile)]
    config_options={'saccdirs': [str], 'output_run_dir': 'NONE',
                    'noisesacc_filename': 'NONE', 'tracers': [str]}

    def get_output_fname(self,name,ext=None):
        fname=self.output_dir+name
        if ext is not None:
            fname+='.'+ext
        return fname

    def parse_input(self) :
        """
        Check sanity of input parameters.
        """
        # This is a hack to get the path of the root output directory.
        # It should be easy to get this from ceci, but I don't know how to.
        self.output_dir = self.get_output('dummy',final_name=True)[:-5]
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        return

    def get_cl_cpld(self, cl, ls_th, leff_hi, wsp_hi, msk_prod):

        cl_mc = wsp_hi.couple_cell(ls_th, cl)[0]/ np.mean(msk_prod)
        cl_intp = scipy.interpolate.interp1d(leff_hi, cl_mc, bounds_error=False,
                       fill_value=(cl_mc[0], cl_mc[-1]))
        cl_o = cl_intp(ls_th)

        return cl_o

    def get_masks(self):

        masks = []
        for trc in self.config['tracers']:
            trc_id, trc_ind = trc.split('_')
            trc_ind = int(trc_ind)

            if trc_id == 'gc':
                fsk, mp_depth = read_flat_map(self.get_input("depth_map"),i_map=0)
                mp_depth[np.isnan(mp_depth)] = 0
                mp_depth[mp_depth > 40] = 0
                msk_depth = np.zeros_like(mp_depth)
                msk_depth[mp_depth >= self.config['depth_cut']] = 1
                fskb, mskfrac = read_flat_map(self.get_input("masked_fraction"), i_map=0)
                # Create binary mask (fraction>threshold and depth req.)
                msk_bo = np.zeros_like(mskfrac)
                msk_bo[mskfrac > self.config['mask_thr']] = 1
                msk_bi = msk_bo * msk_depth
                mask = mskfrac * msk_bi
            elif trc_id == 'wl':
                hdul = fits.open(self.get_input('gamma_maps'))
                _, mask = read_flat_map(None, hdu=[hdul[6 * trc_ind + 2]])
            elif trc_id == 'kappa':
                hdul = fits.open(self.get_input('act_maps'))
                _, mask = read_flat_map(None, hdu=hdul[3])
            elif trc_id == 'y':
                hdul = fits.open(self.get_input('act_maps'))
                _, mask = read_flat_map(None, hdu=hdul[1])
            else:
                raise NotImplementedError()

            masks.append(mask)

        fsk, _ = read_flat_map(self.get_input("masked_fraction"), i_map=0)

        return masks, fsk

    def guess_spectra(self, params, saccfile_coadd, noise_saccfile_coadd):

        if 'dcpl_cl' in self.config.keys():
            logger.info('dcpl_cl provided.')
            if self.config['dcpl_cl']:
                logger.info('Computing coupled guess spectra.')
                saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec = self.guess_spectra_cpld(params,
                                                                                saccfile_coadd, noise_saccfile_coadd)
            else:
                logger.info('Computing uncoupled guess spectra.')
                saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec = self.guess_spectra_uncpld(params,
                                                                                saccfile_coadd, noise_saccfile_coadd)
        else:
            logger.info('dcpl_cl not provided. Computing uncoupled guess spectra.')
            saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec = self.guess_spectra_uncpld(params,
                                                                                saccfile_coadd, noise_saccfile_coadd)

        return saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec

    def guess_spectra_uncpld(self, params, saccfile_coadd, noise_saccfile_coadd=None):

        theor = GSKYPrediction(saccfile_coadd)

        cl_theor = theor.get_prediction(params)

        saccfile_guess_spec = copy.deepcopy(saccfile_coadd)
        if noise_saccfile_coadd is not None:
            saccfile_guess_spec.mean = noise_saccfile_coadd.mean + cl_theor
        else:
            saccfile_guess_spec.mean = cl_theor

        return saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec

    def guess_spectra_cpld(self, params, saccfile_coadd, noise_saccfile_coadd=None):

        ell_theor = np.arange(self.config['ellmax'])
        theor = GSKYPrediction(saccfile_coadd, ells=ell_theor)

        cl_theor = theor.get_prediction(params)

        masks, fsk = self.get_masks()

        dl_min = int(min(2 * np.pi / np.radians(fsk.lx), 2 * np.pi / np.radians(fsk.ly)))
        ells_hi = np.arange(2, 15800, dl_min * 1.5).astype(int)
        bpws_hi = nmt.NmtBinFlat(ells_hi[:-1], ells_hi[1:])
        leff_hi = bpws_hi.get_effective_ells()

        cl_cpld = []
        trc_combs = saccfile_coadd.get_tracer_combinations()
        for i, (tr_i, tr_j) in enumerate(trc_combs):
            tr_i_ind = self.config['tracers'].index(tr_i)
            tr_j_ind = self.config['tracers'].index(tr_j)

            mask_i = masks[tr_i_ind]
            mask_j = masks[tr_j_ind]

            cl_theor_curr = [cl_theor[i]]
            if 'wl' in tr_i:
                field_i = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                            mask_i.reshape([fsk.ny,fsk.nx]),
                            [mask_i.reshape([fsk.ny,fsk.nx]), mask_i.reshape([fsk.ny,fsk.nx])],
                            templates=None)
                cl_theor_curr.append(np.zeros_like(cl_theor[i]))
            else:
                field_i = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                            mask_i.reshape([fsk.ny,fsk.nx]),
                            [mask_i.reshape([fsk.ny,fsk.nx])],
                            templates=None)
            if 'wl' in tr_j:
                field_j = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                            mask_j.reshape([fsk.ny,fsk.nx]),
                            [mask_j.reshape([fsk.ny,fsk.nx]), mask_j.reshape([fsk.ny,fsk.nx])],
                            templates=None)
                cl_theor_curr.append(np.zeros_like(cl_theor[i]))
            else:
                field_j = nmt.NmtFieldFlat(np.radians(fsk.lx), np.radians(fsk.ly),
                            mask_j.reshape([fsk.ny,fsk.nx]),
                            [mask_j.reshape([fsk.ny,fsk.nx])],
                            templates=None)

            wsp_hi_curr = nmt.NmtWorkspaceFlat()
            wsp_hi_curr.compute_coupling_matrix(field_i, field_j, bpws_hi)

            msk_prod = mask_i*mask_j

            cl_cpld_curr = self.get_cl_cpld(cl_theor_curr, ell_theor, leff_hi, wsp_hi_curr, msk_prod)

            if noise_saccfile_coadd is not None:
                if tr_i == tr_j:
                    if 'wl' in tr_i:
                        datatype = 'cl_ee'
                    else:
                        datatype = 'cl_00'
                    l_curr, nl_curr = noise_saccfile_coadd.get_ell_cl(datatype, tr_i, tr_j, return_cov=False)
                    nl_curr_int = scipy.interpolate.interp1d(l_curr, nl_curr, bounds_error=False,
                                                             fill_value=(nl_curr[0], nl_curr[-1]))
                    nl_curr_hi = nl_curr_int(ell_theor)
                    cl_cpld_curr += nl_curr_hi

            cl_cpld.append(cl_cpld_curr)

        # Add tracers to sacc
        saccfile_guess_spec = sacc.Sacc()
        for trc_name, trc in saccfile_coadd.tracers.items():
            saccfile_guess_spec.add_tracer_object(trc)

        for i, (tr_i, tr_j) in trc_combs:
            if 'wl' not in tr_i and 'wl' not in tr_j:
                saccfile_guess_spec.add_ell_cl('cl_00', tr_i, tr_j, ell_theor, cl_cpld[i])
            elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
                saccfile_guess_spec.add_ell_cl('cl_0e', tr_i, tr_j, ell_theor, cl_cpld[i])
                saccfile_guess_spec.add_ell_cl('cl_0b', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))
            else:
                saccfile_guess_spec.add_ell_cl('cl_ee', tr_i, tr_j, ell_theor, cl_cpld[i])
                saccfile_guess_spec.add_ell_cl('cl_eb', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))
                saccfile_guess_spec.add_ell_cl('cl_bb', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))

        return saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec

    def run(self):

        self.parse_input()

        saccfiles = []
        for saccdir in self.config['saccdirs']:
            if self.config['output_run_dir'] != 'NONE':
                path2sacc = os.path.join(saccdir, self.config['output_run_dir'] + '/' + 'power_spectra_wodpj')
            else:
                path2sacc = os.path.join(saccdir, 'power_spectra_wodpj')
            sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
            logger.info('Read {}.'.format(self.get_output_fname(path2sacc, 'sacc')))
            assert sacc_curr.covariance is not None, 'saccfile {} does not contain covariance matrix. Aborting.'.format(
                self.get_output_fname(path2sacc, 'sacc'))
            saccfiles.append(sacc_curr)

        if self.config['noisesacc_filename'] != 'NONE':
            logger.info('Adding noise to theoretical cls.')
            noise_saccfiles = []
            for i, saccdir in enumerate(self.config['saccdirs']):
                if self.config['output_run_dir'] != 'NONE':
                    path2sacc = os.path.join(saccdir, self.config['output_run_dir'] + '/' + self.config['noisesacc_filename'])
                else:
                    path2sacc = os.path.join(saccdir, self.config['noisesacc_filename'])
                noise_sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
                logger.info('Read {}.'.format(self.get_output_fname(path2sacc, 'sacc')))
                if noise_sacc_curr.covariance is None:
                    logger.info('noise sacc has no covariance. Adding covariance matrix to noise sacc.')
                    noise_sacc_curr.add_covariance(saccfiles[i].covariance.covmat)
                noise_saccfiles.append(noise_sacc_curr)
            noise_saccfile_coadd = sutils.coadd_sacc_means(noise_saccfiles, self.config)
        else:
            logger.info('Creating noise-free theoretical cls.')
            noise_saccfile_coadd = None

        # Need to coadd saccfiles after adding covariance to noise saccfiles
        saccfile_coadd = sutils.coadd_sacc_means(saccfiles, self.config)

        params = {
                    'cosmo': self.config['cosmo'],
                    'hmparams': self.config['hmparams']
                  }

        saccfile_coadd, noise_saccfile_coadd, saccfile_guess_spec = self.guess_spectra(params, saccfile_coadd,
                                                                                       noise_saccfile_coadd)

        if self.config['output_run_dir'] != 'NONE':
            input_dir = os.path.join('inputs', self.config['output_run_dir'])
            input_dir = self.get_output_fname(input_dir)
        else:
            input_dir = self.get_output_fname('inputs')
        if not os.path.isdir(input_dir):
            os.makedirs(input_dir)
            logger.info(('Created {}.'.format(input_dir)))

        if self.config['output_run_dir'] != 'NONE':
            coadd_dir = os.path.join('coadds', self.config['output_run_dir'])
            coadd_dir = self.get_output_fname(coadd_dir)
        else:
            coadd_dir = self.get_output_fname('coadds')
        if not os.path.isdir(coadd_dir):
            os.makedirs(coadd_dir)
            logger.info(('Created {}.'.format(coadd_dir)))

        saccfile_coadd.save_fits(os.path.join(coadd_dir, 'saccfile_coadd_test.sacc'), overwrite=True)
        logger.info('Written {}.'.format(os.path.join(coadd_dir, 'saccfile_coadd.sacc')))
        if self.config['noisesacc_filename'] != 'NONE':
            noise_saccfile_coadd.save_fits(os.path.join(coadd_dir, 'noise_saccfile_coadd_test.sacc'), overwrite=True)
            logger.info('Written {}.'.format(os.path.join(coadd_dir, 'noise_saccfile_coadd.sacc')))
        if self.config['noisesacc_filename'] != 'NONE':
            saccfile_guess_spec.save_fits(os.path.join(input_dir, 'saccfile_guess_spectra_test.sacc'), overwrite=True)
            logger.info('Written {}.'.format(os.path.join(input_dir, 'saccfile_guess_spectra.sacc')))
        else:
            saccfile_guess_spec.save_fits(os.path.join(input_dir, 'saccfile_noise-free_guess_spectra_test.sacc'), overwrite=True)
            logger.info('Written {}.'.format(os.path.join(input_dir, 'saccfile_noise-free_guess_spectra.sacc')))

if __name__ == '__main__':
    cls = PipelineStage.main()