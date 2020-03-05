from ceci import PipelineStage
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sacc
from .types import FitsFile, NpyFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

colors = ['#e3a19c', '#85a1ca', '#596d82', '#725e9c', '#3d306b', '#AE7182']

class PSpecPlotter(PipelineStage) :
    name="PSpecPlotter"
    inputs=[]
    outputs=[]
    config_options={'saccdirs': [str], 'output_run_dir': 'NONE', 'output_plot_dir': 'NONE', 'output_dir': 'NONE',
                    'noisesaccs': 'NONE', 'fig_name': str, 'tracers': [str], 'plot_comb': 'all', 'cl_type': 'cl_ee',
                    'plot_errors': False, 'plot_theory': False, 'weightpow': 2, 'logscale_x': False, 'logscale_y': False}

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
            output_dir = os.path.join(self.config['output_dir'], self.config['output_plot_dir'])
        if self.config['output_run_dir'] != 'NONE':
            output_dir = os.path.join(output_dir, self.config['output_run_dir'])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        return

    def plot_spectra(self, saccfile, ntracers, plot_pairs, noise_saccfile=None):

        weightpow = self.config['weightpow']

        indices = []
        for i in range(ntracers):
            for ii in range(i + 1):
                ind = (i, ii)
                indices.append(ind)

        fig = plt.figure(figsize=(44, 32))
        gs = gridspec.GridSpec(ntracers, ntracers)

        for i, (tr_i, tr_j) in enumerate(plot_pairs):

            ax = plt.subplot(gs[indices[i][0], indices[i][1]])

            if self.config['plot_errors']:
                ell_curr, cl_curr = saccfile.get_ell_cl(self.config['cl_type'], tr_j, tr_i, return_cov=True)
            else:
                ell_curr, cl_curr = saccfile.get_ell_cl(self.config['cl_type'], tr_j, tr_i, return_cov=False)

            if noise_saccfile is not None:
                if tr_i == tr_j:
                    ell_curr, cl_noise_curr = noise_saccfile.get_ell_cl(self.config['cl_type'], tr_j, tr_i, return_cov=False)
                    cl_curr -= cl_noise_curr

            if self.config['plot_errors']:
                ax.errorbar(ell_curr, cl_curr * np.power(ell_curr, weightpow), yerr=errs[tbin] * np.power(ell, weightpow),
                            color=colors[3], linestyle='None', marker='.', markersize=15, elinewidth=2.4, capthick=2.4, capsize=3.5,
                            label=r'$C_{\ell}^{%i%i}$' % (tr_i + 1, tr_j + 1))
            else:
                ax.plot(ell_curr, cl_curr * np.power(ell_curr, weightpow), linestyle='None', marker='o', markeredgecolor=colors[3],
                        color=colors[3], label=r'$C_{\ell}^{%i%i}$' % (tr_i, tr_j))
            if self.config['plot_theory']:
                if tr_i == 0 and tr_j == 0:
                    ax.plot(ell_theor, cls_theor * np.power(ell_theor, weightpow), color=colors[-1], \
                            label=r'$\mathrm{pred.}$', lw=2.4, zorder=-32, linestyle='--')

                else:
                    ax.plot(ell_theor, cls_theor * np.power(ell_theor, weightpow), color=colors[-1], lw=2.4, zorder=-32)

            ax.set_xlabel(r'$\ell$')
            if weightpow == 0:
                elltext = ''
            elif weightpow == 1:
                elltext = r'$\ell$'
            else:
                elltext = r'$\ell^{{{}}}$'.format(weightpow)
            ax.set_ylabel(elltext + r'$C_{\ell}$')

            if tr_i == 0 and tr_j == 0:
                handles, labels = ax.get_legend_handles_labels()

                handles = [handles[1], handles[0]]
                labels = [labels[1], labels[0]]

                ax.legend(handles, labels, loc='best', prop={'size': 35})
            else:
                ax.legend(loc='best', prop={'size': 35})
            ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='both')

            if self.config['logscale_x']:
                ax.set_xscale('log')
            if self.config['logscale_y']:
                ax.set_yscale('log')

        if self.config['path2fig'] != 'NONE':
            plt.savefig(self.config['path2fig'], bbox_inches="tight")

            return

    def coadd_saccs(self, saccfiles):

        for i, saccfile in enumerate(saccfiles):
            if not any('y_' in s for s in self.config['tracers']) and not any('kappa_' in s for s in self.config['tracers']):
                if any('y_' in key for key in saccfile.tracers.keys()):
                    for t in saccfile.tracers:
                        if t != 'y_0':
                            saccfile.remove_selection(tracers=('y_0', t))
                if any('kappa_' in key for key in saccfile.tracers.keys()):
                    for t in saccfile.tracers:
                        if t != 'kappa_0':
                            saccfile.remove_selection(tracers=('kappa_0', t))
            if i == 0:
                coadd_mean = saccfile.mean
            else:
                coadd_mean += saccfile.mean

        # Copy sacc
        saccfile_coadd = saccfiles[0].copy()
        # Set mean of new saccfile to coadded mean
        saccfile_coadd.mean(coadd_mean)

        return saccfile_coadd

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
            saccfiles.append(sacc.Sacc.load_fits(self.get_output_fname(path2sacc, 'sacc')))
        saccfile_coadd = self.coadd_saccs(saccfiles)

        if self.config['noisesaccs'] != 'NONE':
            noise_saccfiles = [sacc.Sacc.load_fits(path2sacc) for path2sacc in self.config['path2noisesaccs']]
            noise_saccfile_coadd = self.coadd_saccs(noise_saccfiles)
        else:
            noise_saccfile_coadd = None

        tracer_list = self.config['tracers']
        ntracers = len(tracer_list)

        plot_pairs = []
        if self.config['plot_comb'] == 'all':
            logger.info('Plotting auto- and cross-correlations of tracers.')
            i = 0
            for tr_i in tracer_list:
                for tr_j in tracer_list[:i+1]:
                    # Generate the appropriate list of tracer combinations to plot
                    plot_pairs.append([tr_i, tr_j])
                i += 1
        elif self.config['plot_comb'] == 'auto':
            logger.info('Plotting auto-correlations of tracers.')
            for tr_i in tracer_list:
                plot_pairs.append([tr_i, tr_i])
        elif self.config['plot_comb'] == 'cross':
            logger.info('Plotting cross-correlations of tracers.')
            i = 0
            for tr_i in tracer_list:
                for tr_j in tracer_list[:i]:
                    # Generate the appropriate list of tracer combinations to plot
                    plot_pairs.append([tr_i, tr_j])
                i += 1

        self.plot_spectra(saccfile_coadd, ntracers, plot_pairs, noise_saccfile=noise_saccfile_coadd)

if __name__ == '__main__':
    cls = PipelineStage.main()

