from ceci import PipelineStage
import logging
import numpy as np
import os
import sacc
from .types import FitsFile, DummyFile
from gsky.flatmaps import read_flat_map
from gsky.sims_gauss.MockSurvey import MockSurvey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseMocks(PipelineStage) :
    name="NoiseMocks"
    inputs=[('clean_catalog', FitsFile), ('masked_fraction', FitsFile), ('gamma_maps', FitsFile)]
    outputs=[('dummy', DummyFile)]
    config_options={'probes': ['gamma'], 'spins': [2], 'nrealiz': 1000,
    'path2cls': 'NONE', 'ell_bpws': [100.0,200.0,300.0,
                                     400.0,600.0,800.0,
                                     1000.0,1400.0,1800.0,
                                     2200.0,3000.0,3800.0,
                                     4600.0,6200.0,7800.0,
                                     9400.0,12600.0,15800.0],
    'pixwindow': 0, 'nell_theor': 5000, 'noisemodel': 'data',
    'posfromshearcat': 1, 'shearrot': 'flipqu'}

    def get_output_fname(self, name, ext=None):
        if ext is not None:
            fname = name+'.'+ext
        else:
            fname = name
        fname = os.path.join(self.output_dir, fname)
        return fname

    def parse_input(self) :
        """
        Check sanity of input parameters.
        """
        # This is a hack to get the path of the root output directory.
        # It should be easy to get this from ceci, but I don't know how to.
        self.output_dir = self.get_output('dummy', final_name=True)[:-5]
        if self.config['output_run_dir'] != 'NONE':
            self.output_dir = os.path.join(self.output_dir, self.config['output_run_dir'])
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if self.config['path2theorycls'] != 'NONE':
            assert self.get_output('cls_signal_realiz') != 'NONE', 'Signal cls requested but path2theorycls not provided. Aborting.'

        return

    def cl_realiz_arr_to_sacc(self, cl_realiz_arr, tracer_types, sacc_template):

        logger.info('Creating saccfile for simulated noise.')

        cl_mean = np.mean(cl_realiz_arr, axis=2)

        cl_sacc = sacc_template.copy()
        cl_sacc_mean = np.zeros_like(sacc_template.mean)

        spin2_trcs = [1 for tr in tracer_types if 'wl_' in tr]
        nspin2 = sum(spin2_trcs)
        logger.info('Number of spin 2 fields in tracer array = {}.'.format(nspin2))

        for i, tr in enumerate(tracer_types):
            if 'wl_' in tr:
                ind_ee = cl_sacc.indices(data_type='cl_ee', tracers=(tr, tr))
                ind_bb = cl_sacc.indices(data_type='cl_bb', tracers=(tr, tr))
                cl_sacc_mean[ind_ee] = cl_mean[i, i, :]
                cl_sacc_mean[ind_bb] = cl_mean[i+nspin2, i+nspin2, :]
            if 'gc_' in tr:
                ind_00 = cl_sacc.indices(data_type='cl_00', tracers=(tr, tr))
                cl_sacc_mean[ind_00] = cl_mean[i, i, :]

        # Set mean of new saccfile to coadded mean
        cl_sacc.mean = cl_sacc_mean

        return cl_sacc

    def run(self):
        """
        Main routine. This stage:
        - Creates gamma1, gamma2 maps and corresponding masks from the reduced catalog for a set of redshift bins.
        - Stores the above into a single FITS file.
        """

        self.parse_input()

        logger.info('Running {} realizations of noise power spectra.'.format(self.config['nrealiz']))
        nprobes = len(self.config['probes'])
        logger.info('Number of probes = {}.'.format(nprobes))

        logger.info("Reading masks from {}.".format(self.get_input('gamma_maps')))
        # Here assuming for simplicity that masks are the same
        masks = []
        for i in self.config['ntomo_bins']:
            fsk_temp, mask_temp = read_flat_map(self.get_input('gamma_maps'), i_map=6*i+3)
            mask_temp = mask_temp.reshape([fsk_temp.ny, fsk_temp.nx])
            masks.append(mask_temp)
        masks += [mask_temp]*(nprobes*(nprobes-1)//2)

        if 'spins' in self.config:
            self.config['spins'] = np.array(self.config['spins'])

        noiseparams_keys = ['probes', 'noisemodel', 'posfromshearcat', 'shearrot']
        noiseparams = {key: self.config[key] for key in noiseparams_keys}
        noiseparams['path2shearcat'] = self.get_input('clean_catalog')
        noiseparams['path2fsk'] = self.get_input('masked_fraction')
        simparams_keys = ['probes', 'spins', 'path2theorycls', 'nrealiz', 'ell_bpws', 'pixwindow', 'nell_theor']
        simparams = {key: self.config[key] for key in simparams_keys}
        simparams['path2fsk'] = self.get_input('masked_fraction')

        mocksurvey = MockSurvey(masks, simparams, noiseparams)

        cls, noisecls, ells, wsps = mocksurvey.reconstruct_cls_parallel()

        if os.path.isfile(self.get_output_fname('noi_bias',ext='sacc')):
            logger.info('Reading template sacc from {}.'.format(self.get_output_fname('noi_bias', ext='sacc')))
            sacc_template = sacc.Sacc.load_fits(self.get_output_fname('noi_bias', ext='sacc'))
        else:
            raise RuntimeError('Need template sacc file for analytic noise.')

        noise_sacc = self.cl_realiz_arr_to_sacc(noisecls, self.config['tracers'], sacc_template)

        if self.config['path2cls'] != 'NONE':
            np.save(self.get_output_fname('cls_signal_realiz', 'npy'), cls)
            logger.info('Written signal cls to {}.'.format(self.get_output_fname('cls_signal_realiz', ext='npy')))

        np.save(self.get_output_fname('cls_noise_realiz', 'npy'), noisecls)
        logger.info('Written noise realization cls to {}.'.format(self.get_output_fname('cls_noise_realiz', ext='npy')))

        np.save(self.get_output_fname('l_eff_noise', 'npy'), ells)
        logger.info('Written ells to {}.'.format(self.get_output_fname('l_eff_noise', ext='npy')))

        noise_sacc.save_fits(self.get_output_fname('noi_bias_sim', ext='sacc'), overwrite=True)
        logger.info('Written mean noise cls to {}.'.format(self.get_output_fname('noi_bias_sim', ext='sacc')))

        nprobes = len(self.config['probes'])
        for i in range(nprobes):
            for ii in range(i + 1):
                path2wsp = self.get_output_fname('wsp_probe1={}_probe2={}.dat'.format(i, ii))
                wsps[i][ii].write_to(str(path2wsp))
                logger.info('Written wsp for probe1 = {} and probe2 = {} to {}.'.format(i, ii, path2wsp))

if __name__ == '__main__':
    cls = PipelineStage.main()

