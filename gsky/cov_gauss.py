from ceci import PipelineStage
from .types import FitsFile,ASCIIFile,SACCFile,DummyFile
import numpy as np
import pymaster as nmt
from .power_specter import PowerSpecter
import os
import sacc

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO: Names of files to read
#TODO: COSMOS nz for shear weights

class CovGauss(PowerSpecter) :
    name="CovGauss"
    inputs=[('masked_fraction',FitsFile),('ngal_maps',FitsFile),('shear_maps',FitsFile),
            ('act_maps', FitsFile), ('dust_map',FitsFile),('star_map',FitsFile),
            ('depth_map',FitsFile),('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),
            ('exptime_maps',FitsFile),('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),
            ('seeing_maps',FitsFile),('ellipt_maps',FitsFile),('nvisit_maps',FitsFile),
            ('cosmos_weights',FitsFile),('syst_masking_file',ASCIIFile)]
    outputs=[('dummy',DummyFile)]
    config_options={'ell_bpws':[100.0,200.0,300.0,
                                400.0,600.0,800.0,
                                1000.0,1400.0,1800.0,
                                2200.0,3000.0,3800.0,
                                4600.0,6200.0,7800.0,
                                9400.0,12600.0,15800.0],
                    'oc_dpj_list': ['airmass','seeing','sigma_sky'],
                    'depth_cut':24.5,'band':'i','mask_thr':0.5,'guess_spectrum':'NONE',
                    'gaus_covar_type':'analytic','oc_all_bands':True,
                    'mask_systematics':False,'noise_bias_type':'analytic',
                    'output_run_dir': 'NONE','sys_collapse_type':'average'}

    def get_covar(self, lth, clth, bpws, tracers, wsp, temps, cl_dpj_all):
        """
        Estimate the power spectrum covariance
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param bpws: NaMaster bandpowers.
        :params tracers: tracers.
        :param wsp: NaMaster workspace.
        :param temps: list of contaminant templates.
        :params cl_dpj_all: list of deprojection biases for each bin pair combination.
        """
        if self.config['gaus_covar_type'] == 'analytic':
            print("Computing analytical Gaussian covariance")
            cov = self.get_covar_analytic(lth, clth, bpws, tracers, wsp)
        elif self.config['gaus_covar_type'] == 'gaus_sim':
            print("Computing simulated Gaussian covariance")
            cov = self.get_covar_gaussim(lth, clth, bpws, wsp, temps, cl_dpj_all)

        return cov

    def get_covar_mcm(self, tracers, bpws, tracerCombInd=None):
        """
        Get NmtCovarianceWorkspaceFlat for our mask
        """

        logger.info("Computing covariance MCM.")

        cwsp = [[[[0 for i in range(self.ntracers)] for ii in range(self.ntracers)]
                 for j in range(self.ntracers)] for jj in range(self.ntracers)]

        tracer_combs = []
        for i1 in range(self.ntracers):
            for j1 in range(i1, self.ntracers):
                tracer_combs.append((i1, j1))

        tracer_type_arr = [tr.type for tr in tracers]

        tracerCombInd_curr = 0
        for k1, tup1 in enumerate(tracer_combs):
            tr_i1, tr_j1 = tup1
            for tr_i2, tr_j2 in tracer_combs[k1:]:
                # Check if we need to compute this cwsp
                if tracerCombInd is not None:
                    if tracerCombInd_curr == tracerCombInd:
                        logger.info('tracerCombInd = {}.'.format(tracerCombInd))
                        logger.info('Computing cwsp for tracers = {}, {}.'.format((tr_i1, tr_j1), (tr_i2, tr_j2)))
                        do_cwsp = True
                    else:
                        do_cwsp = False
                else:
                    do_cwsp = True

                if do_cwsp:
                    # File does not exist
                    if not os.path.isfile(
                            self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(tr_i1, tr_j1, tr_i2, tr_j2) + '.dat'):
                        tr_types_cur = np.array(
                            [tracers[tr_i1].type, tracers[tr_j1].type, tracers[tr_i2].type, tracers[tr_j2].type])

                        # All galaxy maps
                        if set(tr_types_cur) == {'delta_g', 'delta_g', 'delta_g', 'delta_g'}:
                            if not hasattr(self, 'cwsp_counts'):
                                counts_indx = tracer_type_arr.index('delta_g')
                                if not os.path.isfile(
                                        self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(counts_indx, counts_indx,
                                                                                              counts_indx,
                                                                                              counts_indx) + '.dat'):
                                    # Compute wsp for counts (is always the same as mask is the same)
                                    self.cwsp_counts = nmt.NmtCovarianceWorkspaceFlat()
                                    logger.info("Computing covariance MCM for counts.")
                                    self.cwsp_counts.compute_coupling_coefficients(tracers[0].field, tracers[0].field, bpws)
                                    self.cwsp_counts.write_to(
                                        self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(counts_indx, counts_indx,
                                                                                              counts_indx,
                                                                                              counts_indx) + '.dat')
                                    logger.info("Covariance MCM written to {}.".format(
                                        self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(counts_indx, counts_indx,
                                                                                              counts_indx,
                                                                                              counts_indx) + '.dat'))
                                else:
                                    logger.info("Reading covariance MCM for counts.")
                                    self.cwsp_counts = nmt.NmtCovarianceWorkspaceFlat()
                                    self.cwsp_counts.read_from(
                                        self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(counts_indx, counts_indx,
                                                                                              counts_indx,
                                                                                              counts_indx) + '.dat')
                                    logger.info("Covariance MCM read from {}.".format(
                                        self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(counts_indx, counts_indx,
                                                                                              counts_indx,
                                                                                              counts_indx) + '.dat'))

                            cwsp_curr = self.cwsp_counts

                        # At least one galaxy map
                        elif 'delta_g' in tr_types_cur:
                            counts_indx = tracer_type_arr.index('delta_g')
                            i1_curr = tr_i1
                            j1_curr = tr_j1
                            i2_curr = tr_i2
                            j2_curr = tr_j2
                            if tracers[tr_i1].type == 'delta_g':
                                i1_curr = counts_indx
                            if tracers[tr_j1].type == 'delta_g':
                                j1_curr = counts_indx
                            if tracers[tr_i2].type == 'delta_g':
                                i2_curr = counts_indx
                            if tracers[tr_j2].type == 'delta_g':
                                j2_curr = counts_indx
                            cwsp_curr = nmt.NmtCovarianceWorkspaceFlat()
                            if not os.path.isfile(
                                    self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(i1_curr, j1_curr, i2_curr,
                                                                                          j2_curr) + '.dat'):
                                # Compute wsp for counts (is always the same as mask is the same)
                                logger.info("Computing covariance MCM for counts xcorr.")
                                cwsp_curr.compute_coupling_coefficients(tracers[0].field, tracers[0].field, bpws)
                                cwsp_curr.write_to(
                                    self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(i1_curr, j1_curr, i2_curr,
                                                                                          j2_curr) + '.dat')
                                logger.info("Covariance MCM written to {}.".format(
                                    self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(i1_curr, j1_curr, i2_curr,
                                                                                          j2_curr) + '.dat'))
                            else:
                                logger.info("Reading covariance MCM for counts xcorr.")
                                cwsp_curr.read_from(
                                    self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(i1_curr, j1_curr, i2_curr,
                                                                                          j2_curr) + '.dat')
                                logger.info("Covariance MCM read from {}.".format(
                                    self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(i1_curr, j1_curr, i2_curr,
                                                                                          j2_curr) + '.dat'))

                        # No galaxy maps
                        else:
                            logger.info("Computing covariance MCM for {}.".format(self.get_output_fname('cov_mcm') +
                                                            '_{}{}{}{}'.format(tr_i1, tr_j1, tr_i2, tr_j2) + '.dat'))
                            cwsp_curr = nmt.NmtCovarianceWorkspaceFlat()
                            cwsp_curr.compute_coupling_coefficients(tracers[tr_i1].field, tracers[tr_j1].field, bpws,
                                                                    tracers[tr_i2].field, tracers[tr_j2].field, bpws)
                            # Write to file
                            cwsp_curr.write_to(
                                self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(tr_i1, tr_j1, tr_i2, tr_j2) + '.dat')

                    # File exists
                    else:
                        logger.info("Reading covariance MCM for {}.".format(
                            self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(tr_i1, tr_j1, tr_i2, tr_j2) + '.dat'))
                        cwsp_curr = nmt.NmtCovarianceWorkspaceFlat()
                        cwsp_curr.read_from(
                            self.get_output_fname('cov_mcm') + '_{}{}{}{}'.format(tr_i1, tr_j1, tr_i2, tr_j2) + '.dat')

                    cwsp[tr_i1][tr_j1][tr_i2][tr_j2] = cwsp_curr

                tracerCombInd_curr += 1

        return cwsp

    def get_covar_gaussim(self, lth, clth, bpws, wsp, temps, cl_dpj_all):
        """
        Estimate the power spectrum covariance from Gaussian simulations
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param bpws: NaMaster bandpowers.
        :param wsp: NaMaster workspace.
        :param temps: list of contaminatn templates.
        :params cl_dpj_all: list of deprojection biases for each bin pair combination.
        """
        # Create a dummy file for the covariance MCM
        f = open(self.get_output_fname('cov_mcm', ext='dat'), "w")
        f.close()

        # Setup
        nsims = 10 * self.ncross * self.nell
        print("Computing covariance from %d Gaussian simulations" % nsims)
        msk_binary = self.msk_bi.reshape([self.fsk.ny, self.fsk.nx])
        weights = (self.msk_bi * self.mskfrac).reshape([self.fsk.ny, self.fsk.nx])
        if temps is not None:
            conts = [[t.reshape([self.fsk.ny, self.fsk.nx])] for t in temps]
            cl_dpj = [[c] for c in cl_dpj_all]
        else:
            conts = None
            cl_dpj = [None for i in range(self.ncross)]

        # Iterate
        cells_sims = []
        for isim in range(nsims):
            if isim % 100 == 0:
                print(" %d-th sim" % isim)
            # Generate random maps
            mps = nmt.synfast_flat(self.fsk.nx, self.fsk.ny,
                                   np.radians(self.fsk.lx), np.radians(self.fsk.ly),
                                   clth, np.zeros(self.nbins), seed=1000 + isim)
            # Nmt fields
            flds = [nmt.NmtFieldFlat(np.radians(self.fsk.lx), np.radians(self.fsk.ly), weights,
                                     [m], templates=conts) for m in mps]
            # Compute power spectra (possibly with deprojection)
            i_x = 0
            cells_this = []
            for i in range(self.nbins):
                for j in range(i, self.nbins):
                    cl = nmt.compute_coupled_cell_flat(flds[i], flds[j], bpws)
                    cells_this.append(wsp.decouple_cell(cl, cl_bias=cl_dpj[i_x])[0])
                    i_x += 1
            cells_sims.append(np.array(cells_this).flatten())
        cells_sims = np.array(cells_sims)
        # Save simulations for further
        np.savez(self.get_output_fname('gaucov_sims', ext='npz'), cl_sims=cells_sims)

        # Compute covariance
        covar = np.cov(cells_sims.T)
        return covar

    def get_covar_analytic(self, lth, clth, bpws, tracers, wsp):
        """
        Estimate the power spectrum covariance analytically
        :param lth: list of multipoles.
        :param clth: list of guess power spectra sampled at the multipoles stored in `lth`.
        :param bpws: NaMaster bandpowers.
        :param tracers: tracers.
        :param wsp: NaMaster workspace.
        """
        # Create a dummy file for the covariance MCM
        f = open(self.get_output_fname('gaucov_analytic', ext='npz'), "w")
        f.close()

        covar = np.zeros([self.ncross, self.nbands, self.ncross, self.nbands])
        # Get covar MCM for counts tracers
        cwsp = self.get_covar_mcm(tracers, bpws)

        tracer_combs = []
        for i1 in range(self.ntracers):
            for j1 in range(i1, self.ntracers):
                tracer_combs.append((i1, j1))

        ix_1 = 0
        for k1, tup1 in enumerate(tracer_combs):
            tr_i1, tr_j1 = tup1
            ix_2 = ix_1
            for tr_i2, tr_j2 in tracer_combs[k1:]:
                ps_inds1 = self.tracers2maps[tr_i1][tr_i2]
                ps_inds2 = self.tracers2maps[tr_i1][tr_j2]
                ps_inds3 = self.tracers2maps[tr_j1][tr_i2]
                ps_inds4 = self.tracers2maps[tr_j1][tr_j2]

                ca1b1 = clth[ps_inds1[:, 0][:4], ps_inds1[:, 1][:4]]
                ca1b2 = clth[ps_inds2[:, 0][:4], ps_inds2[:, 1][:4]]
                ca2b1 = clth[ps_inds3[:, 0][:4], ps_inds3[:, 1][:4]]
                ca2b2 = clth[ps_inds4[:, 0][:4], ps_inds4[:, 1][:4]]

                cov_here = nmt.gaussian_covariance_flat(cwsp[tr_i1][tr_j1][tr_i2][tr_j2], tracers[tr_i1].spin,
                                                        tracers[tr_j1].spin,
                                                        tracers[tr_i2].spin, tracers[tr_j2].spin, lth,
                                                        ca1b1, ca1b2, ca2b1, ca2b2, wsp[tr_i1][tr_j1],
                                                        wsp[tr_i2][tr_j2])

                if set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 0)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 0)):
                    covar[ix_1, :, ix_2, :] = cov_here
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_here.T
                    ix_2 += 1
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 2)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 2)):
                    cov_here = cov_here.reshape([self.nbands, 2, self.nbands, 2])
                    cov_te_te = cov_here[:, 0, :, 0]
                    cov_te_tb = cov_here[:, 0, :, 1]
                    cov_tb_te = cov_here[:, 1, :, 0]
                    cov_tb_tb = cov_here[:, 1, :, 1]

                    covar[ix_1, :, ix_2, :] = cov_te_te
                    covar[ix_1, :, ix_2 + 1, :] = cov_te_tb
                    covar[ix_1 + 1, :, ix_2, :] = cov_tb_te
                    covar[ix_1 + 1, :, ix_2 + 1, :] = cov_tb_tb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_te_te.T
                        covar[ix_2 + 1, :, ix_1, :] = cov_te_tb.T
                        covar[ix_2, :, ix_1 + 1, :] = cov_tb_te.T
                        covar[ix_2 + 1, :, ix_1 + 1, :] = cov_tb_tb.T
                    ix_2 += 2
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 0)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 2)):
                    cov_here = cov_here.reshape([self.nbands, 1, self.nbands, 2])
                    cov_tt_te = cov_here[:, 0, :, 0]
                    cov_tt_tb = cov_here[:, 0, :, 1]

                    covar[ix_1, :, ix_2, :] = cov_tt_te
                    covar[ix_1, :, ix_2 + 1, :] = cov_tt_tb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_tt_te.T
                        covar[ix_2 + 1, :, ix_1, :] = cov_tt_tb.T
                    ix_2 += 2
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 2)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 0)):
                    cov_here = cov_here.reshape([self.nbands, 1, self.nbands, 2])
                    cov_tt_te = cov_here[:, 0, :, 0]
                    cov_tt_tb = cov_here[:, 0, :, 1]

                    covar[ix_1, :, ix_2, :] = cov_tt_te
                    covar[ix_1 + 1, :, ix_2, :] = cov_tt_tb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_tt_te.T
                        covar[ix_2, :, ix_1 + 1, :] = cov_tt_tb.T
                    ix_2 += 1
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 0)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((2, 2)):
                    cov_here = cov_here.reshape([self.nbands, 1, self.nbands, 4])
                    cov_tt_ee = cov_here[:, 0, :, 0]
                    cov_tt_eb = cov_here[:, 0, :, 1]
                    cov_tt_be = cov_here[:, 0, :, 2]
                    cov_tt_bb = cov_here[:, 0, :, 3]

                    covar[ix_1, :, ix_2, :] = cov_tt_ee
                    covar[ix_1, :, ix_2 + 1, :] = cov_tt_eb
                    covar[ix_1, :, ix_2 + 2, :] = cov_tt_be
                    covar[ix_1, :, ix_2 + 3, :] = cov_tt_bb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_tt_ee.T
                        covar[ix_2 + 1, :, ix_1, :] = cov_tt_eb.T
                        covar[ix_2 + 2, :, ix_1, :] = cov_tt_be.T
                        covar[ix_2 + 3, :, ix_1, :] = cov_tt_bb.T
                    ix_2 += 4
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((2, 2)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 0)):
                    cov_here = cov_here.reshape([self.nbands, 1, self.nbands, 4])
                    cov_tt_ee = cov_here[:, 0, :, 0]
                    cov_tt_eb = cov_here[:, 0, :, 1]
                    cov_tt_be = cov_here[:, 0, :, 2]
                    cov_tt_bb = cov_here[:, 0, :, 3]

                    covar[ix_1, :, ix_2, :] = cov_tt_ee
                    covar[ix_1 + 2, :, ix_2, :] = cov_tt_eb
                    covar[ix_1 + 2, :, ix_2, :] = cov_tt_be
                    covar[ix_1 + 3, :, ix_2, :] = cov_tt_bb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_tt_ee.T
                        covar[ix_2, :, ix_1 + 2, :] = cov_tt_eb.T
                        covar[ix_2, :, ix_1 + 2, :] = cov_tt_be.T
                        covar[ix_2, :, ix_1 + 3, :] = cov_tt_bb.T
                    ix_2 += 1
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 2)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((2, 2)):
                    cov_here = cov_here.reshape([self.nbands, 2, self.nbands, 4])
                    cov_te_ee = cov_here[:, 0, :, 0]
                    cov_te_eb = cov_here[:, 0, :, 1]
                    cov_te_be = cov_here[:, 0, :, 2]
                    cov_te_bb = cov_here[:, 0, :, 3]
                    cov_tb_ee = cov_here[:, 1, :, 0]
                    cov_tb_eb = cov_here[:, 1, :, 1]
                    cov_tb_be = cov_here[:, 1, :, 2]
                    cov_tb_bb = cov_here[:, 1, :, 3]

                    covar[ix_1, :, ix_2, :] = cov_te_ee
                    covar[ix_1, :, ix_2 + 1, :] = cov_te_eb
                    covar[ix_1, :, ix_2 + 2, :] = cov_te_be
                    covar[ix_1, :, ix_2 + 3, :] = cov_te_bb
                    covar[ix_1 + 1, :, ix_2, :] = cov_tb_ee
                    covar[ix_1 + 1, :, ix_2 + 1, :] = cov_tb_eb
                    covar[ix_1 + 1, :, ix_2 + 2, :] = cov_tb_be
                    covar[ix_1 + 1, :, ix_2 + 3, :] = cov_tb_bb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_te_ee.T
                        covar[ix_2 + 1, :, ix_1, :] = cov_te_eb.T
                        covar[ix_2 + 2, :, ix_1, :] = cov_te_be.T
                        covar[ix_2 + 3, :, ix_1, :] = cov_te_bb.T
                        covar[ix_2, :, ix_1 + 1, :] = cov_tb_ee.T
                        covar[ix_2 + 1, :, ix_1 + 1, :] = cov_tb_eb.T
                        covar[ix_2 + 2, :, ix_1 + 1, :] = cov_tb_be.T
                        covar[ix_2 + 3, :, ix_1 + 1, :] = cov_tb_bb.T
                    ix_2 += 4
                elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((2, 2)) and set(
                        (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 2)):
                    cov_here = cov_here.reshape([self.nbands, 2, self.nbands, 4])
                    cov_te_ee = cov_here[:, 0, :, 0]
                    cov_te_eb = cov_here[:, 0, :, 1]
                    cov_te_be = cov_here[:, 0, :, 2]
                    cov_te_bb = cov_here[:, 0, :, 3]
                    cov_tb_ee = cov_here[:, 1, :, 0]
                    cov_tb_eb = cov_here[:, 1, :, 1]
                    cov_tb_be = cov_here[:, 1, :, 2]
                    cov_tb_bb = cov_here[:, 1, :, 3]

                    covar[ix_1, :, ix_2, :] = cov_te_ee
                    covar[ix_1 + 1, :, ix_2, :] = cov_te_eb
                    covar[ix_1 + 2, :, ix_2, :] = cov_te_be
                    covar[ix_1 + 3, :, ix_2, :] = cov_te_bb
                    covar[ix_1, :, ix_2 + 1, :] = cov_tb_ee
                    covar[ix_1 + 1, :, ix_2 + 1, :] = cov_tb_eb
                    covar[ix_1 + 2, :, ix_2 + 1, :] = cov_tb_be
                    covar[ix_1 + 3, :, ix_2 + 1, :] = cov_tb_bb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_te_ee.T
                        covar[ix_2, :, ix_1 + 1, :] = cov_te_eb.T
                        covar[ix_2, :, ix_1 + 2, :] = cov_te_be.T
                        covar[ix_2, :, ix_1 + 3, :] = cov_te_bb.T
                        covar[ix_2 + 1, :, ix_1, :] = cov_tb_ee.T
                        covar[ix_2 + 1, :, ix_1 + 1, :] = cov_tb_eb.T
                        covar[ix_2 + 1, :, ix_1 + 2, :] = cov_tb_be.T
                        covar[ix_2 + 1, :, ix_1 + 3, :] = cov_tb_bb.T
                    ix_2 += 2
                else:
                    cov_here = cov_here.reshape([self.nbands, 4, self.nbands, 4])
                    cov_ee_ee = cov_here[:, 0, :, 0]
                    cov_ee_eb = cov_here[:, 0, :, 1]
                    cov_ee_be = cov_here[:, 0, :, 2]
                    cov_ee_bb = cov_here[:, 0, :, 3]
                    cov_eb_ee = cov_here[:, 1, :, 0]
                    cov_eb_eb = cov_here[:, 1, :, 1]
                    cov_eb_be = cov_here[:, 1, :, 2]
                    cov_eb_bb = cov_here[:, 1, :, 3]
                    cov_be_ee = cov_here[:, 2, :, 0]
                    cov_be_eb = cov_here[:, 2, :, 1]
                    cov_be_be = cov_here[:, 2, :, 2]
                    cov_be_bb = cov_here[:, 2, :, 3]
                    cov_bb_ee = cov_here[:, 3, :, 0]
                    cov_bb_eb = cov_here[:, 3, :, 1]
                    cov_bb_be = cov_here[:, 3, :, 2]
                    cov_bb_bb = cov_here[:, 3, :, 3]

                    covar[ix_1, :, ix_2, :] = cov_ee_ee
                    covar[ix_1, :, ix_2 + 1, :] = cov_ee_eb
                    covar[ix_1, :, ix_2 + 2, :] = cov_ee_be
                    covar[ix_1, :, ix_2 + 3, :] = cov_ee_bb
                    covar[ix_1 + 1, :, ix_2, :] = cov_eb_ee
                    covar[ix_1 + 1, :, ix_2 + 1, :] = cov_eb_eb
                    covar[ix_1 + 1, :, ix_2 + 2, :] = cov_eb_be
                    covar[ix_1 + 1, :, ix_2 + 3, :] = cov_eb_bb
                    covar[ix_1 + 2, :, ix_2, :] = cov_be_ee
                    covar[ix_1 + 2, :, ix_2 + 1, :] = cov_be_eb
                    covar[ix_1 + 2, :, ix_2 + 2, :] = cov_be_be
                    covar[ix_1 + 2, :, ix_2 + 3, :] = cov_be_bb
                    covar[ix_1 + 3, :, ix_2, :] = cov_bb_ee
                    covar[ix_1 + 3, :, ix_2 + 1, :] = cov_bb_eb
                    covar[ix_1 + 3, :, ix_2 + 2, :] = cov_bb_be
                    covar[ix_1 + 3, :, ix_2 + 3, :] = cov_bb_bb
                    if (tr_i1, tr_j1) != (tr_i2, tr_j2):
                        covar[ix_2, :, ix_1, :] = cov_ee_ee.T
                        covar[ix_2 + 1, :, ix_1, :] = cov_ee_eb.T
                        covar[ix_2 + 2, :, ix_1, :] = cov_ee_be.T
                        covar[ix_2 + 3, :, ix_1, :] = cov_ee_bb.T
                        covar[ix_2, :, ix_1 + 1, :] = cov_eb_ee.T
                        covar[ix_2 + 1, :, ix_1 + 1, :] = cov_eb_eb.T
                        covar[ix_2 + 2, :, ix_1 + 1, :] = cov_eb_be.T
                        covar[ix_2 + 3, :, ix_1 + 1, :] = cov_eb_bb.T
                        covar[ix_2, :, ix_1 + 2, :] = cov_be_ee.T
                        covar[ix_2 + 1, :, ix_1 + 2, :] = cov_be_eb.T
                        covar[ix_2 + 2, :, ix_1 + 2, :] = cov_be_be.T
                        covar[ix_2 + 3, :, ix_1 + 2, :] = cov_be_bb.T
                        covar[ix_2, :, ix_1 + 3, :] = cov_bb_ee.T
                        covar[ix_2 + 1, :, ix_1 + 3, :] = cov_bb_eb.T
                        covar[ix_2 + 2, :, ix_1 + 3, :] = cov_bb_be.T
                        covar[ix_2 + 3, :, ix_1 + 3, :] = cov_bb_bb.T
                    ix_2 += 4
            if set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 0)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 0)):
                ix_1 += 1

            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 0)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 2)):
                ix_1 += 1
            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 2)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 0)):
                ix_1 += 2

            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 2)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 2)):
                ix_1 += 2

            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 0)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((2, 2)):
                ix_1 += 1
            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((2, 2)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 0)):
                ix_1 += 4

            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((0, 2)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((2, 2)):
                ix_1 += 2
            elif set((tracers[tr_i1].spin, tracers[tr_j1].spin)) == set((2, 2)) and set(
                    (tracers[tr_i2].spin, tracers[tr_j2].spin)) == set((0, 2)):
                ix_1 += 4

            else:
                ix_1 += 4

        covar = covar.reshape([self.ncross * self.nbands, self.ncross * self.nbands])

        return covar

    def run(self) :
        """
        Main function.
        This stage:
        - Produces measurements of the power spectrum with and without contaminant deprojections.
        - Estimates the noise bias
        - Estimates the covariance matrix
        - Estimates the deprojection bias
        """
        self.parse_input()

        logger.info("Reading mask.")
        self.msk_bi,self.mskfrac,self.mp_depth=self.get_masks()

        logger.info("Computing area.")
        self.area_pix=np.radians(self.fsk.dx)*np.radians(self.fsk.dy)
        self.area_patch=np.sum(self.msk_bi*self.mskfrac)*self.area_pix
        self.lmax=int(180.*np.sqrt(1./self.fsk.dx**2+1./self.fsk.dy**2))

        logger.info("Reading contaminants.")
        temps=self.get_contaminants()

        logger.info("Setting bandpowers.")
        lini=np.array(self.config['ell_bpws'])[:-1]
        lend=np.array(self.config['ell_bpws'])[ 1:]
        bpws=nmt.NmtBinFlat(lini,lend)
        ell_eff=bpws.get_effective_ells()
        self.nbands = ell_eff.shape[0]
        logger.info('Number of ell bands = {}.'.format(self.nbands))

        tracers_nc, tracers_wc = self.get_all_tracers(temps)

        self.ntracers = len(tracers_nc)
        self.nmaps = self.ntracers_counts + self.ntracers_comptony + 2*self.ntracers_shear

        # Set up mapping
        self.mapping(tracers_nc)

        logger.info("Getting MCM.")
        wsp = self.get_mcm(tracers_nc,bpws)

        self.ncross = self.nmaps * (self.nmaps + 1) // 2 + self.ntracers_shear
        if self.config['gaus_covar_type'] == 'analytic':
            logger.info("Computing analytic covariance.")
            if not os.path.isfile(self.get_input('power_spectra_wdpj')):
                logger.info("Computing deprojected power spectra.")
                logger.info(" W. deprojections.")
                cls_wdpj, _ = self.get_power_spectra(tracers_wc, wsp, bpws)

            else:
                logger.info("Reading deprojected power spectra.")
                s = sacc.Sacc.load_fits(self.get_input('power_spectra_wdpj'))
                cls_wdpj_mean = s.mean
                cls_wdpj = self.convert_sacc_to_clarr(cls_wdpj_mean, tracers_wc)

            logger.info("Getting guess power spectra.")
            lth, clth = self.get_cl_guess(ell_eff, cls_wdpj)

            cov_wodpj = self.get_covar(lth,clth,bpws,tracers_wc,wsp,None,None)
            cov_wdpj = cov_wodpj.copy()

        else:
            logger.info("Computing simulated covariance.")
            if not os.path.isfile(self.get_input('power_spectra_wdpj')):
                logger.info("Computing deprojected power spectra.")
                logger.info(" W. deprojections.")
                cls_wdpj, cls_wdpj_coupled = self.get_power_spectra(tracers_wc, wsp, bpws)

            else:
                logger.info("Reading deprojected power spectra.")
                s = sacc.Sacc.load_fits(self.get_input('power_spectra_wdpj'))
                cls_wdpj_mean = s.mean
                cls_wdpj = self.convert_sacc_to_clarr(cls_wdpj_mean, tracers_wc)
                logger.info("Reading deprojected coupled power spectra.")
                s = sacc.Sacc.load_fits(self.get_input('power_spectra_wdpj_coupled'))
                cls_wdpj_coupled_mean = s.mean
                cls_wdpj_coupled = self.convert_sacc_to_clarr(cls_wdpj_coupled_mean, tracers_wc)

            logger.info("Getting guess power spectra.")
            lth, clth = self.get_cl_guess(ell_eff, cls_wdpj)

            if os.path.isfile(self.get_output_fname('dpj_bias', ext='sacc')):
                s = sacc.Sacc.load_fits(self.get_output_fname('dpj_bias', ext='sacc'))
                cl_deproj_bias_mean = s.mean
                cl_deproj_bias = self.convert_sacc_to_clarr(cl_deproj_bias_mean, tracers_wc)
            else:
                logger.info("Computing deprojection bias.")
                _, cl_deproj_bias = self.get_dpj_bias(tracers_wc, lth, clth, cls_wdpj_coupled, wsp, bpws)

            cov_wodpj = self.get_covar(lth, clth, bpws, tracers_wc, wsp, None, None)
            cov_wdpj=self.get_covar(lth,clth,bpws,tracers_wc,wsp,temps,cl_deproj_bias)

        # Write covariances into existing sacc
        if os.path.isfile(self.get_output_fname('power_spectra_wodpj',ext='sacc')):
            logger.info('{} provided.'.format(self.get_output_fname('power_spectra_wodpj',ext='sacc')))
            logger.info('Adding non deprojected covariance matrix to {}.'.format(self.get_output_fname('power_spectra_wodpj',ext='sacc')))
            s_wodpj = sacc.Sacc.load_fits(self.get_output_fname('power_spectra_wodpj',ext='sacc'))
            s_wodpj.add_covariance(cov_wodpj)
            s_wodpj.save_fits(self.get_output_fname('power_spectra_wodpj',ext='sacc'), overwrite=True)
            logger.info('Written non deprojected covariance matrix.')
        else:
            logger.info('{} not provided.'.format(self.get_output_fname('power_spectra_wodpj',ext='sacc')))
            logger.info('Writing non deprojected covariance matrix to {}.'.format(self.get_output_fname('cov_wodpj',ext='sacc')))
            s_wodpj = sacc.Sacc()
            s_wodpj.add_covariance(cov_wodpj)
            s_wodpj.save_fits(self.get_output_fname('cov_wodpj',ext='sacc'), overwrite=True)
            logger.info('Written non deprojected covariance matrix.')

        if os.path.isfile(self.get_output_fname('power_spectra_wdpj',ext='sacc')):
            logger.info('{} provided.'.format(self.get_output_fname('power_spectra_wdpj',ext='sacc')))
            logger.info('Adding deprojected covariance matrix to {}.'.format(self.get_output_fname('power_spectra_wdpj',ext='sacc')))
            s_wdpj = sacc.Sacc.load_fits(self.get_output_fname('power_spectra_wdpj',ext='sacc'))
            s_wdpj.add_covariance(cov_wdpj)
            s_wdpj.save_fits(self.get_output_fname('power_spectra_wdpj',ext='sacc'), overwrite=True)
            logger.info('Written deprojected covariance matrix.')
        else:
            logger.info('{} not provided.'.format(self.get_output_fname('power_spectra_wdpj',ext='sacc')))
            logger.info('Writing non deprojected covariance matrix to {}.'.format(self.get_output_fname('cov_wdpj',ext='sacc')))
            s_wdpj = sacc.Sacc()
            s_wdpj.add_covariance(cov_wdpj)
            s_wdpj.save_fits(self.get_output_fname('cov_wdpj',ext='sacc'), overwrite=True)
            logger.info('Written deprojected covariance matrix.')

if __name__ == '__main__':
    cls = PipelineStage.main()
