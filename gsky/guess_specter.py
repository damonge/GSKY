from ceci import PipelineStage
import numpy as np
import logging
import os
import copy
import scipy.interpolate
from astropy.io import fits
import pymaster as nmt
from .flatmaps import read_flat_map
from .types import FitsFile, DummyFile, SACCFile
import sacc
from theory.predict_theory import GSKYPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuessSpecter(PipelineStage) :
    name="GuessSpecter"
    inputs=[('masked_fraction', FitsFile), ('depth_map', FitsFile),
            ('gamma_maps', FitsFile), ('act_maps', FitsFile),
            ('saccfile_signal', SACCFile), ('saccfile_noise', SACCFile)]
    outputs=[('dummy', DummyFile)]
    config_options={'cosmo': dict, 'hmparams': dict, 'ellmax': 30000, 'cpl_cl': True}

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
        if self.config['output_run_dir'] != 'NONE':
            self.output_dir += self.config['output_run_dir'] + '/'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        return

    def get_cl_cpld(self, cl, ls_th, leff_hi, wsp_hi, msk_prod):

        cl_mc = wsp_hi.couple_cell(ls_th, cl)[0]/ np.mean(msk_prod)
        cl_intp = scipy.interpolate.interp1d(leff_hi, cl_mc, bounds_error=False,
                       fill_value=(cl_mc[0], cl_mc[-1]))
        cl_o = cl_intp(ls_th)

        return cl_o

    def get_masks(self, tracers):

        logger.info('Reading masks.')

        masks = []
        for trc in tracers:
            logger.info('Reading mask for tracer = {}.'.format(trc))
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

    def guess_spectra(self, params, saccfile_signal, saccfile_noise):

        if 'cpl_cl' in self.config.keys():
            logger.info('cpl_cl provided.')
            if self.config['cpl_cl']:
                logger.info('Computing coupled guess spectra.')
                saccfile_guess_spec = self.guess_spectra_cpld(params, saccfile_signal, saccfile_noise)
            else:
                logger.info('Computing uncoupled guess spectra.')
                saccfile_guess_spec = self.guess_spectra_dcpld(params, saccfile_signal, saccfile_noise)
        else:
            logger.info('dcpl_cl not provided. Computing uncoupled guess spectra.')
            saccfile_guess_spec = self.guess_spectra_dcpld(params, saccfile_signal, saccfile_noise)

        return saccfile_guess_spec

    def guess_spectra_dcpld(self, params, saccfile_signal, saccfile_noise=None):

        ell_theor = np.arange(self.config['ellmax'])
        theor = GSKYPrediction(saccfile_signal, ells=ell_theor)

        cl_theor = theor.get_prediction(params)

        saccfile_guess_spec = copy.deepcopy(saccfile_signal)
        if saccfile_noise is not None:
            saccfile_guess_spec.mean = saccfile_noise.mean + cl_theor
        else:
            saccfile_guess_spec.mean = cl_theor

        return saccfile_guess_spec

    def guess_spectra_cpld(self, params, saccfile_signal, saccfile_noise=None):

        ell_theor = np.arange(self.config['ellmax'])
        theor = GSKYPrediction(saccfile_signal, ells=ell_theor)

        cl_theor = theor.get_prediction(params)

        tracers = list(saccfile_signal.tracers.keys())
        tracer_id_arr = [tr.split('_')[0] for tr in tracers]
        masks, fsk = self.get_masks(tracers)

        dl_min = int(min(2 * np.pi / np.radians(fsk.lx), 2 * np.pi / np.radians(fsk.ly)))
        ells_hi = np.arange(2, 15800, dl_min * 1.5).astype(int)
        bpws_hi = nmt.NmtBinFlat(ells_hi[:-1], ells_hi[1:])
        leff_hi = bpws_hi.get_effective_ells()

        cl_cpld = []
        trc_combs = saccfile_signal.get_tracer_combinations()
        for i, (tr_i, tr_j) in enumerate(trc_combs):

            logger.info('Computing wsp for trc_comb = {}.'.format((tr_i, tr_j)))

            tr_i_id, _ = tr_i.split('_')
            tr_j_id, _ = tr_j.split('_')
            tr_i_ind = tracers.index(tr_i)
            tr_j_ind = tracers.index(tr_j)

            mask_i = masks[tr_i_ind]
            mask_j = masks[tr_j_ind]

            cl_theor_curr = [cl_theor[i]]
            if tr_i_id == 'wl':
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
            if tr_j_id == 'wl':
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

            if tr_i_id == 'wl' and tr_j_id == 'wl':
                cl_theor_curr.append(np.zeros_like(cl_theor[i]))

            # File does not exist
            if not os.path.isfile(self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind, tr_j_ind) + '.dat'):
                # All galaxy maps
                if tr_i_id == 'gc' and tr_j_id == 'gc':
                    if not hasattr(self, 'wsp_counts'):
                        counts_indx = tracer_id_arr.index('gc')
                        wsp_hi_curr = nmt.NmtWorkspaceFlat()
                        if not os.path.isfile(
                                self.get_output_fname('mcm_hi') + '_{}{}'.format(counts_indx, counts_indx) + '.dat'):
                            logger.info("Computing MCM for counts.")
                            wsp_hi_curr.compute_coupling_matrix(field_i, field_j, bpws_hi)
                            wsp_hi_curr.write_to(
                                self.get_output_fname('mcm_hi') + '_{}{}'.format(counts_indx, counts_indx) + '.dat')
                            logger.info("MCM written to {}.".format(
                                self.get_output_fname('mcm_hi') + '_{}{}'.format(counts_indx, counts_indx) + '.dat'))
                        else:
                            logger.info("Reading MCM for counts.")
                            wsp_hi_curr.read_from(
                                self.get_output_fname('mcm_hi') + '_{}{}'.format(counts_indx, counts_indx) + '.dat')
                            logger.info("MCM read from {}.".format(
                                self.get_output_fname('mcm_hi') + '_{}{}'.format(counts_indx, counts_indx) + '.dat'))
                        self.wsp_counts = wsp_hi_curr
                    wsp_hi_curr = self.wsp_counts

                # One galaxy map
                elif tr_i_id == 'gc' or tr_j_id == 'gc':
                    counts_indx = tracer_id_arr.index('gc')
                    tr_i_ind_curr = tr_i_ind
                    tr_j_ind_curr = tr_j_ind
                    if tr_i_id == 'gc':
                        tr_i_ind_curr = counts_indx
                    if tr_j_id == 'gc':
                        tr_j_ind_curr = counts_indx
                    wsp_hi_curr = nmt.NmtWorkspaceFlat()
                    if not os.path.isfile(
                            self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind_curr, tr_j_ind_curr) + '.dat'):
                        logger.info("Computing MCM for counts xcorr.")
                        wsp_hi_curr.compute_coupling_matrix(field_i, field_j, bpws_hi)
                        wsp_hi_curr.write_to(self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind_curr, tr_j_ind_curr) + '.dat')
                        logger.info("MCM written to {}.".format(
                            self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind_curr, tr_j_ind_curr) + '.dat'))
                    else:
                        logger.info("Reading MCM for counts xcorr.")
                        wsp_hi_curr.read_from(
                            self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind_curr, tr_j_ind_curr) + '.dat')
                        logger.info("MCM read from {}.".format(
                            self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind_curr, tr_j_ind_curr) + '.dat'))

                # No galaxy maps
                else:
                    logger.info(
                        "Computing MCM for {}.".format(self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind, tr_j_ind) + '.dat'))
                    wsp_hi_curr = nmt.NmtWorkspaceFlat()
                    wsp_hi_curr.compute_coupling_matrix(field_i, field_j, bpws_hi)
                    wsp_hi_curr.write_to(self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind, tr_j_ind) + '.dat')

            # File exists
            else:
                logger.info("Reading MCM for {}.".format(self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind, tr_j_ind) + '.dat'))
                wsp_hi_curr = nmt.NmtWorkspaceFlat()
                wsp_hi_curr.read_from(self.get_output_fname('mcm_hi') + '_{}{}'.format(tr_i_ind, tr_j_ind) + '.dat')

            msk_prod = mask_i*mask_j

            cl_cpld_curr = self.get_cl_cpld(cl_theor_curr, ell_theor, leff_hi, wsp_hi_curr, msk_prod)

            if saccfile_noise is not None:
                logger.info('Adding noise.')
                if tr_i == tr_j:
                    if 'wl' in tr_i:
                        datatype = 'cl_ee'
                    else:
                        datatype = 'cl_00'
                    l_curr, nl_curr = saccfile_noise.get_ell_cl(datatype, tr_i, tr_j, return_cov=False)
                    nl_curr_int = scipy.interpolate.interp1d(l_curr, nl_curr, bounds_error=False,
                                                             fill_value=(nl_curr[0], nl_curr[-1]))
                    nl_curr_hi = nl_curr_int(ell_theor)
                    cl_cpld_curr += nl_curr_hi
                    if 'wl' in tr_i:
                        l_curr, nl_curr = saccfile_noise.get_ell_cl('cl_bb', tr_i, tr_j, return_cov=False)
                        nl_curr_int = scipy.interpolate.interp1d(l_curr, nl_curr, bounds_error=False,
                                                                 fill_value=(nl_curr[0], nl_curr[-1]))
                        nl_curr_bb_hi = nl_curr_int(ell_theor)

                        cl_cpld_curr = np.vstack((cl_cpld_curr, nl_curr_bb_hi))

            cl_cpld.append(cl_cpld_curr)

        # Add tracers to sacc
        saccfile_guess_spec = sacc.Sacc()
        for trc_name, trc in saccfile_signal.tracers.items():
            saccfile_guess_spec.add_tracer_object(trc)

        for i, (tr_i, tr_j) in enumerate(trc_combs):
            if 'wl' not in tr_i and 'wl' not in tr_j:
                saccfile_guess_spec.add_ell_cl('cl_00', tr_i, tr_j, ell_theor, cl_cpld[i])
            elif ('wl' in tr_i and 'wl' not in tr_j) or ('wl' not in tr_i and 'wl' in tr_j):
                saccfile_guess_spec.add_ell_cl('cl_0e', tr_i, tr_j, ell_theor, cl_cpld[i])
                saccfile_guess_spec.add_ell_cl('cl_0b', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))
            else:
                if tr_i == tr_j:
                    saccfile_guess_spec.add_ell_cl('cl_ee', tr_i, tr_j, ell_theor, cl_cpld[i][0, :])
                    saccfile_guess_spec.add_ell_cl('cl_eb', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i][0, :]))
                    saccfile_guess_spec.add_ell_cl('cl_be', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i][0, :]))
                    saccfile_guess_spec.add_ell_cl('cl_bb', tr_i, tr_j, ell_theor, cl_cpld[i][1, :])
                else:
                    saccfile_guess_spec.add_ell_cl('cl_ee', tr_i, tr_j, ell_theor, cl_cpld[i])
                    saccfile_guess_spec.add_ell_cl('cl_eb', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))
                    saccfile_guess_spec.add_ell_cl('cl_be', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))
                    saccfile_guess_spec.add_ell_cl('cl_bb', tr_i, tr_j, ell_theor, np.zeros_like(cl_cpld[i]))

        return saccfile_guess_spec

    def run(self):

        self.parse_input()

        saccfile_signal = sacc.Sacc.load_fits(self.get_input("saccfile_signal"))
        logger.info('Read {}.'.format(self.get_input("saccfile_signal")))

        if self.get_input("saccfile_noise") != 'NONE':
            logger.info('Adding noise to theoretical cls.')
            saccfile_noise = sacc.Sacc.load_fits(self.get_input("saccfile_noise"))
            logger.info('Read {}.'.format(self.get_input("saccfile_noise")))
        else:
            logger.info('Creating noise-free theoretical cls.')
            saccfile_noise = None

        params = {
                    'cosmo': self.config['cosmo'],
                    'hmparams': self.config['hmparams']
                  }

        saccfile_guess_spec = self.guess_spectra(params, saccfile_signal, saccfile_noise)

        if self.get_input("saccfile_noise") != 'NONE':
            if self.config['cpl_cl']:
                saccfile_guess_spec.save_fits(self.get_output_fname('guess_spectra_cpld', ext='sacc'),
                                              overwrite=True)
                logger.info('Written {}.'.format(self.get_output_fname('guess_spectra_cpld', ext='sacc')))
            else:
                saccfile_guess_spec.save_fits(self.get_output_fname('guess_spectra_dcpld', ext='sacc'),
                                              overwrite=True)
                logger.info('Written {}.'.format(self.get_output_fname('guess_spectra_dcpld', ext='sacc')))
        else:
            if self.config['cpl_cl']:
                saccfile_guess_spec.save_fits(self.get_output_fname('noise-free_guess_spectra_cpld',
                                                                    ext='sacc'), overwrite=True)
                logger.info('Written {}.'.format('noise-free_guess_spectra_cpld', ext='sacc'))
            else:
                saccfile_guess_spec.save_fits(self.get_output_fname('noise-free_guess_spectra_dcpld',
                                                                    ext='sacc'), overwrite=True)
                logger.info('Written {}.'.format('noise-free_guess_spectra_dcpld', ext='sacc'))

        # Permissions on NERSC
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

if __name__ == '__main__':
    cls = PipelineStage.main()