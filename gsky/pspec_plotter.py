from ceci import PipelineStage
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sacc
from theory.predict_theory import GSKYPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

colors = ['#e3a19c', '#85a1ca', '#596d82', '#725e9c', '#3d306b', '#AE7182', 'IndianRed', '#5d61a2']

class PSpecPlotter(PipelineStage) :
    name="PSpecPlotter"
    inputs=[]
    outputs=[]
    config_options={'saccdirs': [str], 'output_run_dir': 'NONE', 'output_plot_dir': 'NONE', 'output_dir': 'NONE',
                    'noisesaccs': 'NONE', 'fig_name': str, 'tracers': [str], 'plot_comb': 'all', 'cl_type': 'cl_ee',
                    'plot_errors': False, 'plot_theory': False, 'weightpow': 2, 'logscale_x': False, 'logscale_y': False,
                    'coadd_noise': False, 'coadd_mode': 'invvar', 'plot_fields': False, 'ell_theor': 'NONE'}

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
        self.output_dir = self.config['output_dir']+'/'
        if self.config['output_plot_dir'] != 'NONE':
            self.output_plot_dir = os.path.join(self.config['output_dir'], self.config['output_plot_dir'])
        if self.config['output_run_dir'] != 'NONE':
            self.output_plot_dir = os.path.join(self.output_plot_dir, self.config['output_run_dir'])
        if not os.path.isdir(self.output_plot_dir):
            os.makedirs(self.output_plot_dir)

        if 'ell_max_trc' in self.config:
            self.ell_max_dict = dict(zip(self.config['tracers'], self.config['ell_max_trc']))
        if 'ell_min_trc' in self.config:
            self.ell_min_dict = dict(zip(self.config['tracers'], self.config['ell_min_trc']))

        return

    def plot_spectra(self, saccfile, ntracers, plot_pairs, noise_saccfile=None, fieldsaccs=None, field_noisesaccs=None,
                     params=None, plot_indx=None):

        if plot_indx is not None:
            weightpow = self.config['weightpow'][plot_indx]
            plot_theory = self.config['plot_theory'][plot_indx]
            plot_comb = self.config['plot_comb'][plot_indx]
            plot_errors = self.config['plot_errors']
            cl_type = self.config['cl_type'][plot_indx]
            ell_theor = self.config['ell_theor']
            logscale_x = self.config['logscale_x'][plot_indx]
            logscale_y = self.config['logscale_y'][plot_indx]
            fig_name = self.config['fig_name'][plot_indx]
        else:
            weightpow = self.config['weightpow']
            plot_theory = self.config['plot_theory']
            plot_comb = self.config['plot_comb']
            plot_errors = self.config['plot_errors']
            cl_type = self.config['cl_type']
            ell_theor = self.config['ell_theor']
            logscale_x = self.config['logscale_x']
            logscale_y = self.config['logscale_y']
            fig_name = self.config['fig_name']

        if plot_theory:
            logger.info('plot_theory = True. Computing theory predictions.')
            theor = GSKYPrediction(saccfile, ell_theor)
            cl_theor = theor.get_prediction(params, trc_combs=plot_pairs)

            # Compute reduced chi2
            indx_temp = np.hstack([saccfile.indices(cl_type, (tr_i_temp, tr_j_temp)) for
                                  (tr_i_temp, tr_j_temp) in plot_pairs])
            cl_theor_temp = cl_theor[indx_temp]
            cl_temp = saccfile.mean[indx_temp]
            delta = cl_temp - cl_theor_temp
            if noise_saccfile is not None:
                delta -= noise_saccfile.mean[indx_temp]
            cov_temp = saccfile.covariance.covmat[np.ix_(indx_temp, indx_temp)]
            invcov = np.linalg.inv(cov_temp)
            chi2_red = np.einsum('i,ij,j', delta, invcov, delta) / (delta.shape[0] - self.config['n_fitparams'])
            logger.info('Reduced chi2 = chi2/dof = {}.'.format(chi2_red))
            logger.info('dof = {}.'.format(delta.shape[0]))

        indices = []
        if plot_comb == 'all':
            for i in range(ntracers):
                for ii in range(i + 1):
                    ind = (i, ii)
                    indices.append(ind)
        elif plot_comb == 'auto':
            for i in range(ntracers):
                ind = (i, i)
                indices.append(ind)
        elif plot_comb == 'cross':
            assert ntracers.shape[0] == 2, 'ntracers required for cross-correlation needs to be 2D. Aborting.'
            for i in range(ntracers[0]):
                for ii in range(ntracers[1]):
                    ind = (i, ii)
                    indices.append(ind)

        if np.atleast_1d(ntracers).shape[0] == 2:
            fig = plt.figure(figsize=(ntracers[1]*11, ntracers[0]*8))
            gs = gridspec.GridSpec(ntracers[0], ntracers[1])
        else:
            fig = plt.figure(figsize=(ntracers * 11, ntracers * 8))
            gs = gridspec.GridSpec(ntracers, ntracers)

        for i, (tr_i, tr_j) in enumerate(plot_pairs):

            ax = plt.subplot(gs[indices[i][0], indices[i][1]])

            if plot_errors:
                ell_curr, cl_curr, cov_curr = saccfile.get_ell_cl(cl_type, tr_i, tr_j, return_cov=True)
                err_curr = np.sqrt(np.diag(cov_curr))
                if np.any(np.isnan(err_curr)):
                    logger.info('Found negative diagonal elements of covariance matrix. Setting to zero.')
                    err_curr[np.isnan(err_curr)] = 0
            else:
                ell_curr, cl_curr = saccfile.get_ell_cl(cl_type, tr_i, tr_j, return_cov=False)

            if noise_saccfile is not None:
                if tr_i == tr_j:
                    ell_curr, cl_noise_curr = noise_saccfile.get_ell_cl(cl_type, tr_i, tr_j, return_cov=False)
                    cl_curr -= cl_noise_curr

            # Plot the mean
            if plot_errors:
                if weightpow != -1:
                    ax.errorbar(ell_curr, cl_curr * np.power(ell_curr, weightpow), yerr=err_curr * np.power(ell_curr, weightpow),
                                color='k', linestyle='--', marker='o', markeredgecolor='k', linewidth=2, markersize=9,
                                elinewidth=2, capthick=2, capsize=3.5, label=r'$C_{{\ell}}^{{{}{}}}$'.format(tr_i, tr_j))
                else:
                    ax.errorbar(ell_curr, cl_curr * ell_curr*(ell_curr+1)/2./np.pi, yerr=err_curr * ell_curr*(ell_curr+1)/2./np.pi,
                                color='k', linestyle='--', marker='o', markeredgecolor='k', linewidth=2, markersize=9,
                                elinewidth=2, capthick=2, capsize=3.5, label=r'$C_{{\ell}}^{{{}{}}}$'.format(tr_i, tr_j))
            else:
                if weightpow != -1:
                    ax.plot(ell_curr, cl_curr * np.power(ell_curr, weightpow), linestyle='--', marker='o', markeredgecolor='k',
                            color='k', label=r'$C_{{\ell}}^{{{}{}}}$'.format(tr_i, tr_j), linewidth=2, markersize=9)
                else:
                    ax.plot(ell_curr, cl_curr * ell_curr*(ell_curr+1)/2./np.pi, linestyle='--', marker='o',
                            markeredgecolor='k',
                            color='k', label=r'$C_{{\ell}}^{{{}{}}}$'.format(tr_i, tr_j), linewidth=2, markersize=9)

            # Now plot the individual fields
            if fieldsaccs is not None:
                for ii, fieldsacc in enumerate(fieldsaccs):
                    ell_field, cl_field = fieldsacc.get_ell_cl(cl_type, tr_i, tr_j, return_cov=False)
                    if field_noisesaccs is not None:
                        _, cl_noise_field = field_noisesaccs[ii].get_ell_cl(cl_type, tr_i, tr_j,
                                                                   return_cov=False)
                        cl_field -= cl_noise_field
                    if indices[i][0] == 0 and indices[i][1] == 0:
                        if weightpow != -1:
                            ax.plot(ell_field, cl_field * np.power(ell_field, weightpow), linestyle='--', marker='o',
                                markeredgecolor=colors[ii], color=colors[ii], zorder=-1, alpha=0.8,
                                label=r'$\mathrm{{{}}}$'.format(self.config['saccdirs'][ii][:-5]))
                        else:
                            ax.plot(ell_field, cl_field * ell_field*(ell_field+1)/2./np.pi, linestyle='--',
                                    marker='o',
                                    markeredgecolor=colors[ii], color=colors[ii], zorder=-1, alpha=0.8,
                                    label=r'$\mathrm{{{}}}$'.format(self.config['saccdirs'][ii][:-5]))
                    else:
                        if weightpow != -1:
                            ax.plot(ell_field, cl_field * np.power(ell_field, weightpow), linestyle='--', marker='o',
                                markeredgecolor=colors[ii], color=colors[ii], zorder=-1, alpha=0.8)
                        else:
                            ax.plot(ell_field, cl_field * ell_field*(ell_field+1)/2./np.pi, linestyle='--',
                                    marker='o', zorder=-1, alpha=0.8,
                                    markeredgecolor=colors[ii], color=colors[ii])
            if plot_theory:
                indx_curr = saccfile.indices(cl_type, (tr_i, tr_j))
                cl_theor_curr = cl_theor[indx_curr]
                if ell_theor == 'NONE':
                    ell = ell_curr
                else:
                    ell = ell_theor
                if indices[i][0] == 0 and indices[i][1] == 0:
                    if weightpow != -1:
                        ax.plot(ell, cl_theor_curr * np.power(ell, weightpow), color=colors[-1], \
                                label=r'$\mathrm{pred.}$', lw=2.4, zorder=-32)
                    else:
                        ax.plot(ell, cl_theor_curr * ell*(ell+1)/2./np.pi, color=colors[-1], \
                                label=r'$\mathrm{pred.}$', lw=2.4, zorder=-32)

                else:
                    if weightpow != -1:
                        ax.plot(ell, cl_theor_curr * np.power(ell, weightpow), color=colors[-1], lw=2.4, zorder=-32)
                    else:
                        ax.plot(ell, cl_theor_curr * ell*(ell+1)/2./np.pi, color=colors[-1], lw=2.4,
                                zorder=-32)

                delta = cl_curr - cl_theor_curr
                invcov = np.linalg.inv(cov_curr)
                chi2_red = np.einsum('i,ij,j', delta, invcov, delta) / (delta.shape[0] - self.config['n_fitparams'])
                logger.info('{} {}: Reduced chi2 = chi2/dof = {}.'.format(tr_i, tr_j, chi2_red))
                logger.info('dof = {}.'.format(delta.shape[0]))

            ax.set_xlabel(r'$\ell$')
            if weightpow == 0:
                elltext = ''
            elif weightpow == 1:
                elltext = r'$\ell$'
            elif weightpow == -1:
                elltext = r'$\ell (\ell+1)/(2\pi)$'
            else:
                elltext = r'$\ell^{{{}}}$'.format(weightpow)
            ax.set_ylabel(elltext + r'$C_{\ell}$')

            if indices[i][0] == 0 and indices[i][1] == 0:
                # handles, labels = ax.get_legend_handles_labels()
                #
                # handles = [handles[1], handles[0]]
                # labels = [labels[1], labels[0]]
                #
                # ax.legend(handles, labels, loc='best', prop={'size': 16}, ncol=2, frameon=False)
                ax.legend(loc='best', prop={'size': 16}, ncol=2, frameon=False)
            else:
                ax.legend(loc='best', prop={'size': 16}, frameon=False)
            ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='both')

            if logscale_x:
                ax.set_xscale('log')
            if logscale_y:
                ax.set_yscale('log')

        if fig_name != 'NONE':
            logger.info('Saving figure to {}.'.format(os.path.join(self.output_plot_dir, fig_name)))
            plt.savefig(os.path.join(self.output_plot_dir, fig_name), bbox_inches="tight")

            return

    def coadd_saccs(self, saccfiles, is_noisesacc=False):

        if self.config['coadd_mode'] == 'invvar':
            saccfile_coadd = self.coadd_saccs_invvar(saccfiles)
        elif self.config['coadd_mode'] == 'separate':
            saccfile_coadd = self.coadd_saccs_separate(saccfiles, is_noisesacc)
        else:
            raise NotImplementedError('Only coadd_mode = invvar and separate implemented. Aborting.')

        return saccfile_coadd

    def coadd_saccs_invvar(self, saccfiles):

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

            if self.ell_max_dict is not None:
                logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
                for tr_i, tr_j in saccfile.get_tracer_combinations():
                    if tr_i in self.config['tracers'] and tr_j in self.config['tracers']:
                        ell_max_curr = min(self.ell_max_dict[tr_i], self.ell_max_dict[tr_j])
                        logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
                        saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
                    else:
                        saccfile.remove_selection(tracers=(tr_i, tr_j))
                logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

            if self.ell_min_dict is not None:
                logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
                for tr_i, tr_j in saccfile.get_tracer_combinations():
                    if tr_i in self.config['tracers'] and tr_j in self.config['tracers']:
                        ell_min_curr = max(self.ell_min_dict[tr_i], self.ell_min_dict[tr_j])
                        logger.info('Removing ells < {} for {}, {}.'.format(ell_min_curr, tr_i, tr_j))
                        saccfile.remove_selection(tracers=(tr_i, tr_j), ell__lt=ell_min_curr)
                    else:
                        saccfile.remove_selection(tracers=(tr_i, tr_j))
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

        assert set(self.config['tracers']) <= set(tempsacc_tracers), 'Larger tracer set requested than present in largest ' \
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

    def coadd_saccs_separate(self, saccfiles, is_noisesacc=False):

        logger.info('Coadding saccfiles with common probes.')

        for i, saccfile in enumerate(saccfiles):
            if not any('y_' in s for s in self.config['tracers']) and not any('kappa_' in s for s in self.config['tracers']):
                if any('y_' in key for key in saccfile.tracers.keys()):
                    for t in saccfile.tracers:
                        logger.info('Removing y_0 from {}.'.format(self.config['saccdirs'][i]))
                        saccfile.remove_selection(tracers=('y_0', t))
                        saccfile.remove_selection(tracers=(t, 'y_0'))
                if any('kappa_' in key for key in saccfile.tracers.keys()):
                    for t in saccfile.tracers:
                        logger.info('Removing kappa_0 from {}.'.format(self.config['saccdirs'][i]))
                        saccfile.remove_selection(tracers=('kappa_0', t))
                        saccfile.remove_selection(tracers=(t, 'kappa_0'))

            if self.ell_max_dict is not None:
                logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
                for tr_i, tr_j in saccfile.get_tracer_combinations():
                    if tr_i in self.config['tracers'] and tr_j in self.config['tracers']:
                        ell_max_curr = min(self.ell_max_dict[tr_i], self.ell_max_dict[tr_j])
                        logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
                        saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
                    else:
                        saccfile.remove_selection(tracers=(tr_i, tr_j))
                logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

            if self.ell_min_dict is not None:
                logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
                for tr_i, tr_j in saccfile.get_tracer_combinations():
                    if tr_i in self.config['tracers'] and tr_j in self.config['tracers']:
                        ell_min_curr = max(self.ell_min_dict[tr_i], self.ell_min_dict[tr_j])
                        logger.info('Removing ells < {} for {}, {}.'.format(ell_min_curr, tr_i, tr_j))
                        saccfile.remove_selection(tracers=(tr_i, tr_j), ell__lt=ell_min_curr)
                    else:
                        saccfile.remove_selection(tracers=(tr_i, tr_j))
                logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

            if i == 0:
                coadd_mean = saccfile.mean
                if self.config['plot_errors']:
                    if not is_noisesacc:
                        coadd_cov = saccfile.covariance.covmat
                    else:
                        if self.config['coadd_noise']:
                            coadd_cov = saccfile.covariance.covmat
            else:
                coadd_mean += saccfile.mean
                if self.config['plot_errors']:
                    if not is_noisesacc:
                        coadd_cov += saccfile.covariance.covmat
                    else:
                        if self.config['coadd_noise']:
                            coadd_cov += saccfile.covariance.covmat

        n_saccs = len(saccfiles)
        coadd_mean /= n_saccs
        if self.config['plot_errors']:
            if not is_noisesacc:
                coadd_cov /= n_saccs ** 2
            else:
                if self.config['coadd_noise']:
                    coadd_cov /= n_saccs ** 2

        # Copy sacc
        saccfile_coadd = saccfiles[0].copy()
        # Set mean of new saccfile to coadded mean
        saccfile_coadd.mean = coadd_mean
        if self.config['plot_errors']:
            if not is_noisesacc:
                saccfile_coadd.add_covariance(coadd_cov)
            else:
                if self.config['coadd_noise']:
                    saccfile_coadd.add_covariance(coadd_cov)

        return saccfile_coadd

    def run(self):
        """
        Main routine. This stage:
        - Creates gamma1, gamma2 maps and corresponding masks from the reduced catalog for a set of redshift bins.
        - Stores the above into a single FITS file.
        """

        self.parse_input()
        if 'theory' in self.config.keys():
            logger.info('theory config provided.')
            theory_params = self.config['theory']
        else:
            logger.info('No theory config provided.')
            theory_params = None

        saccfiles = []
        for saccdir in self.config['saccdirs']:
            if self.config['output_run_dir'] != 'NONE':
                path2sacc = os.path.join(saccdir, self.config['output_run_dir']+'/'+'power_spectra_wodpj')
            sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
            logger.info('Read {}.'.format(self.get_output_fname(path2sacc, 'sacc')))
            if self.config['plot_errors']:
                assert sacc_curr.covariance is not None, \
                    'plot_errors = True but saccfiles {} does not contain covariance matrix. Aborting.'.format(self.get_output_fname(path2sacc, 'sacc'))
            saccfiles.append(sacc_curr)

        if self.config['noisesacc_filename'] != 'NONE':
            logger.info('Reading provided noise saccfile.')
            noise_saccfiles = []
            for i, saccdir in enumerate(self.config['saccdirs']):
                if self.config['output_run_dir'] != 'NONE':
                    path2sacc = os.path.join(saccdir, self.config['output_run_dir'] + '/' + self.config['noisesacc_filename'])
                noise_sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
                if self.config['plot_errors']:
                    if self.config['coadd_noise']:
                        assert noise_sacc_curr.covariance is not None, \
                            'plot_errors = True but noise saccfile {} does not contain covariance matrix. Aborting.'.format(self.get_output_fname(path2sacc, 'sacc'))
                    else:
                        if self.config['coadd_mode'] == 'invvar':
                            logger.info('coadd_mode = invvar. Adding covariance matrix to noise sacc.')
                            noise_sacc_curr.add_covariance(saccfiles[i].covariance.covmat)
                noise_saccfiles.append(noise_sacc_curr)
            noise_saccfile_coadd = self.coadd_saccs(noise_saccfiles, is_noisesacc=True)
        else:
            logger.info('No noise saccfile provided.')
            noise_saccfile_coadd = None
            noise_saccfiles = None

        saccfile_coadd = self.coadd_saccs(saccfiles)

        if type(self.config['cl_type']) is list:
            logger.info('Generating list of plots.')

            for pl_indx in range(len(self.config['cl_type'])):
                plot_tracer_list = self.config['plot_tracers'][pl_indx]
                ntracers = len(plot_tracer_list)

                plot_pairs = []
                if self.config['plot_comb'][pl_indx] == 'all':
                    logger.info('Plotting auto- and cross-correlations of tracers.')
                    i = 0
                    for tr_i in plot_tracer_list:
                        for tr_j in plot_tracer_list[:i+1]:
                            # Generate the appropriate list of tracer combinations to plot
                            plot_pairs.append([tr_j, tr_i])
                        i += 1
                elif self.config['plot_comb'][pl_indx] == 'auto':
                    logger.info('Plotting auto-correlations of tracers.')
                    for tr_i in plot_tracer_list:
                        plot_pairs.append([tr_i, tr_i])
                elif self.config['plot_comb'][pl_indx] == 'cross':
                    tracer_type_list = [tr.split('_')[0] for tr in plot_tracer_list]
                    # Get unique tracers and keep ordering
                    unique_trcs = []
                    [unique_trcs.append(tr) for tr in tracer_type_list if tr not in unique_trcs]
                    ntracers0 = tracer_type_list.count(unique_trcs[0])
                    ntracers1 = tracer_type_list.count(unique_trcs[1])
                    ntracers = np.array([ntracers0, ntracers1])
                    logger.info('Plotting cross-correlations of tracers.')
                    i = 0
                    for tr_i in plot_tracer_list[:ntracers0]:
                        for tr_j in plot_tracer_list[ntracers0:]:
                            if tr_i.split('_')[0] != tr_j.split('_')[0]:
                                # Generate the appropriate list of tracer combinations to plot
                                plot_pairs.append([tr_i, tr_j])
                        i += 1
                else:
                    raise NotImplementedError('Only plot_comb = all, auto and cross supported. Aborting.')

                if not self.config['plot_fields']:
                    logger.info('Not plotting single fields.')
                    saccfiles = None
                    noise_saccfiles = None
                else:
                    logger.info('Plotting single fields.')

                logger.info('Plotting tracer combination = {}.'.format(plot_pairs))

                self.plot_spectra(saccfile_coadd, ntracers, plot_pairs, noise_saccfile=noise_saccfile_coadd, fieldsaccs=saccfiles,
                                  field_noisesaccs=noise_saccfiles, params=theory_params, plot_indx=pl_indx)

        else:
            logger.info('Generating only single plot.')

            plot_tracer_list = self.config['plot_tracers']
            ntracers = len(plot_tracer_list)

            plot_pairs = []
            if self.config['plot_comb'] == 'all':
                logger.info('Plotting auto- and cross-correlations of tracers.')
                i = 0
                for tr_i in plot_tracer_list:
                    for tr_j in plot_tracer_list[:i + 1]:
                        # Generate the appropriate list of tracer combinations to plot
                        plot_pairs.append([tr_j, tr_i])
                    i += 1
            elif self.config['plot_comb'] == 'auto':
                logger.info('Plotting auto-correlations of tracers.')
                for tr_i in plot_tracer_list:
                    plot_pairs.append([tr_i, tr_i])
            elif self.config['plot_comb'] == 'cross':
                tracer_type_list = [tr.split('_')[0] for tr in plot_tracer_list]
                # Get unique tracers and keep ordering
                unique_trcs = []
                [unique_trcs.append(tr) for tr in tracer_type_list if tr not in unique_trcs]
                ntracers0 = tracer_type_list.count(unique_trcs[0])
                ntracers1 = tracer_type_list.count(unique_trcs[1])
                ntracers = np.array([ntracers0, ntracers1])
                logger.info('Plotting cross-correlations of tracers.')
                i = 0
                for tr_i in plot_tracer_list[:ntracers0]:
                    for tr_j in plot_tracer_list[ntracers0:]:
                        if tr_i.split('_')[0] != tr_j.split('_')[0]:
                            # Generate the appropriate list of tracer combinations to plot
                            plot_pairs.append([tr_i, tr_j])
                    i += 1
            else:
                raise NotImplementedError('Only plot_comb = all, auto and cross supported. Aborting.')

            if not self.config['plot_fields']:
                logger.info('Not plotting single fields.')
                saccfiles = None
                noise_saccfiles = None
            else:
                logger.info('Plotting single fields.')

            logger.info('Plotting tracer combination = {}.'.format(plot_pairs))

            self.plot_spectra(saccfile_coadd, ntracers, plot_pairs, noise_saccfile=noise_saccfile_coadd,
                              fieldsaccs=saccfiles,
                              field_noisesaccs=noise_saccfiles, params=theory_params)

        # Permissions on NERSC
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

if __name__ == '__main__':
    cls = PipelineStage.main()

