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
                    'coadd_noise': False}

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

        return

    def plot_spectra(self, saccfile, ntracers, plot_pairs, noise_saccfile=None, fieldsaccs=None, field_noisesaccs=None, params=None):

        weightpow = self.config['weightpow']

        if self.config['plot_theory']:
            logger.info('plot_theory = True. Computing theory predictions.')
            ell_theor, _ = saccfile.get_ell_cl(self.config['cl_type'], plot_pairs[0][0], plot_pairs[0][1], return_cov=False)
            theor = GSKYPrediction(saccfile, ell_theor)
            cl_theor = theor.get_prediction(params)

        indices = []
        if self.config['plot_comb'] == 'all':
            for i in range(ntracers):
                for ii in range(i + 1):
                    ind = (i, ii)
                    indices.append(ind)
        elif self.config['plot_comb'] == 'auto':
            for i in range(ntracers):
                ind = (i, i)
                indices.append(ind)
        elif self.config['plot_comb'] == 'cross':
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

            if self.config['plot_errors']:
                ell_curr, cl_curr, cov_curr = saccfile.get_ell_cl(self.config['cl_type'], tr_i, tr_j, return_cov=True)
                err_curr = np.sqrt(np.diag(cov_curr))
                if np.any(np.isnan(err_curr)):
                    logger.info('Found negative diagonal elements of covariance matrix. Setting to zero.')
                    err_curr[np.isnan(err_curr)] = 0
            else:
                ell_curr, cl_curr = saccfile.get_ell_cl(self.config['cl_type'], tr_i, tr_j, return_cov=False)

            if noise_saccfile is not None:
                if tr_i == tr_j:
                    ell_curr, cl_noise_curr = noise_saccfile.get_ell_cl(self.config['cl_type'], tr_i, tr_j, return_cov=False)
                    cl_curr -= cl_noise_curr

            # Plot the mean
            if self.config['plot_errors']:
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
                    ell_field, cl_field = fieldsacc.get_ell_cl(self.config['cl_type'], tr_i, tr_j, return_cov=False)
                    if field_noisesaccs is not None:
                        _, cl_noise_field = field_noisesaccs[ii].get_ell_cl(self.config['cl_type'], tr_i, tr_j,
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
            if self.config['plot_theory']:
                indx_curr = saccfile.indices(self.config['cl_type'], (tr_i, tr_j))
                cl_theor_curr = cl_theor[indx_curr]
                if indices[i][0] == 0 and indices[i][1] == 0:
                    if weightpow != -1:
                        ax.plot(ell_theor, cl_theor_curr * np.power(ell_theor, weightpow), color=colors[-1], \
                                label=r'$\mathrm{pred.}$', lw=2.4, zorder=-32)
                    else:
                        ax.plot(ell_theor, cl_theor_curr * ell_theor*(ell_theor+1)/2./np.pi, color=colors[-1], \
                                label=r'$\mathrm{pred.}$', lw=2.4, zorder=-32)

                else:
                    if weightpow != -1:
                        ax.plot(ell_theor, cl_theor_curr * np.power(ell_theor, weightpow), color=colors[-1], lw=2.4, zorder=-32)
                    else:
                        ax.plot(ell_theor, cl_theor_curr * ell_theor*(ell_theor+1)/2./np.pi, color=colors[-1], lw=2.4,
                                zorder=-32)

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

            if self.config['logscale_x']:
                ax.set_xscale('log')
            if self.config['logscale_y']:
                ax.set_yscale('log')

        if self.config['fig_name'] != 'NONE':
            logger.info('Saving figure to {}.'.format(os.path.join(self.output_plot_dir, self.config['fig_name'])))
            plt.savefig(os.path.join(self.output_plot_dir, self.config['fig_name']), bbox_inches="tight")

            return

    def coadd_saccs(self, saccfiles, is_noisesacc=False):

        logger.info('Coadding saccfiles.')

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
        saccfile_coadd = self.coadd_saccs(saccfiles)

        if self.config['noisesacc_filename'] != 'NONE':
            logger.info('Reading provided noise saccfile.')
            noise_saccfiles = []
            for saccdir in self.config['saccdirs']:
                if self.config['output_run_dir'] != 'NONE':
                    path2sacc = os.path.join(saccdir, self.config['output_run_dir'] + '/' + self.config['noisesacc_filename'])
                noise_sacc_curr = sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc'))
                if self.config['plot_errors']:
                    if self.config['coadd_noise']:
                        assert noise_sacc_curr.covariance is not None, \
                            'plot_errors = True but noise saccfile {} does not contain covariance matrix. Aborting.'.format(self.get_output_fname(path2sacc, 'sacc'))
                noise_saccfiles.append(noise_sacc_curr)
            noise_saccfile_coadd = self.coadd_saccs(noise_saccfiles, is_noisesacc=True)
        else:
            logger.info('No noise saccfile provided.')
            noise_saccfile_coadd = None
            noise_saccfiles = None

        tracer_list = self.config['tracers']
        ntracers = len(tracer_list)

        plot_pairs = []
        if self.config['plot_comb'] == 'all':
            logger.info('Plotting auto- and cross-correlations of tracers.')
            i = 0
            for tr_i in tracer_list:
                for tr_j in tracer_list[:i+1]:
                    # Generate the appropriate list of tracer combinations to plot
                    plot_pairs.append([tr_j, tr_i])
                i += 1
        elif self.config['plot_comb'] == 'auto':
            logger.info('Plotting auto-correlations of tracers.')
            for tr_i in tracer_list:
                plot_pairs.append([tr_i, tr_i])
        elif self.config['plot_comb'] == 'cross':
            tracer_type_list = [tr.split('_')[0] for tr in tracer_list]
            # Get unique tracers and keep ordering
            unique_trcs = []
            [unique_trcs.append(tr) for tr in tracer_type_list if tr not in unique_trcs]
            ntracers0 = tracer_type_list.count(unique_trcs[0])
            ntracers1 = tracer_type_list.count(unique_trcs[1])
            ntracers = np.array([ntracers0, ntracers1])
            logger.info('Plotting cross-correlations of tracers.')
            i = 0
            for tr_i in tracer_list[:ntracers0]:
                for tr_j in tracer_list[ntracers0:]:
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
                          field_noisesaccs=noise_saccfiles, params=theory_params)

if __name__ == '__main__':
    cls = PipelineStage.main()

