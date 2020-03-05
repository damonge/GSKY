from ceci import PipelineStage
import logging
import numpy as np
import os
from .types import FitsFile, NpyFile
from gsky.flatmaps import read_flat_map
from gsky.sims_gauss.MockSurvey import MockSurvey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseMocks(PipelineStage) :
    name="NoiseMocks"
    inputs=[('calib_catalog', FitsFile), ('masked_fraction', FitsFile), ('gamma_maps', FitsFile)]
    outputs=[('cls_noise_realiz', NpyFile), ('l_eff_noise', NpyFile)]
    config_options={'probes': ['gamma'], 'spins': [2], 'nrealiz': 1000,
    'path2cls': 'NONE', 'ell_bpws': [100.0,200.0,300.0,
                                     400.0,600.0,800.0,
                                     1000.0,1400.0,1800.0,
                                     2200.0,3000.0,3800.0,
                                     4600.0,6200.0,7800.0,
                                     9400.0,12600.0,15800.0],
    'pixwindow': 0, 'nell_theor': 5000, 'noisemodel': 'data',
    'posfromshearcat': 1, 'shearrot': 'flipu'}

    def get_output_fname(self, name, ext=None):
        fname = self.output_dir+name
        if ext is not None:
            fname += '.'+ext
        return fname

    def parse_input(self) :
        """
        Check sanity of input parameters.
        """
        # This is a hack to get the path of the root output directory.
        # It should be easy to get this from ceci, but I don't know how to.
        self.output_dir = os.path.dirname(self.get_output('cls_noise_realiz'))
        if self.config['output_run_dir'] != 'NONE':
            self.output_dir += self.config['output_run_dir']+'/'
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        if self.config['path2theorycls'] != 'NONE':
            assert self.get_output('cls_signal_realiz') != 'NONE', 'Signal cls requested but path2theorycls not provided. Aborting.'

        return

    def run(self):
        """
        Main routine. This stage:
        - Creates gamma1, gamma2 maps and corresponding masks from the reduced catalog for a set of redshift bins.
        - Stores the above into a single FITS file.
        """

        self.parse_input()

        logger.info('Running {} realizations of noise power spectra.'.format(self.config['nrealiz']))

        logger.info("Reading masks from {}.".format(self.get_input('gamma_maps')))
        # Here assuming for simplicity that masks are the same
        masks = []
        for i in self.config['ntomo_bins']:
            fsk_temp, mask_temp = read_flat_map(self.get_input('gamma_maps'), i_map=6*i+3)
            mask_temp = mask_temp.reshape([fsk_temp.ny, fsk_temp.nx])
            masks.append(mask_temp)

        if 'spins' in self.config:
            self.config['spins'] = np.array(self.config['spins'])

        noiseparams_keys = ['probes', 'noisemodel', 'posfromshearcat', 'shearrot']
        noiseparams = {key: self.config[key] for key in noiseparams_keys}
        noiseparams['path2shearcat'] = self.get_input('calib_catalog')
        noiseparams['path2fsk'] = self.get_input('masked_fraction')
        simparams_keys = ['probes', 'spins', 'path2theorycls', 'nrealiz', 'ell_bpws', 'pixwindow', 'nell_theor']
        simparams = {key: self.config[key] for key in simparams_keys}
        simparams['path2fsk'] = self.get_input('masked_fraction')

        mocksurvey = MockSurvey(masks, simparams, noiseparams)

        cls, noisecls, ells, wsps = mocksurvey.reconstruct_cls_parallel()

        if self.config['path2cls'] != 'NONE':
            np.save(self.get_output('cls_signal_realiz'), cls)
            logger.info('Written signal cls to {}.'.format(self.get_output('cls_signal_realiz')))

        np.save(self.get_output('cls_noise_realiz'), noisecls)
        logger.info('Written noise cls to {}.'.format(self.get_output('cls_noise_realiz')))

        np.save(self.get_output('l_eff_noise'), ells)
        logger.info('Written ells to {}.'.format(self.get_output('l_eff_noise')))

        for i in range(self.config['nprobes']):
            for ii in range(i + 1):
                path2wsp = self.get_output_fname('wsp_probe1={}_probe2={}.dat'.format(i, ii))
                wsps[i][ii].write_to(str(path2wsp))
                logger.info('Written wsp for probe1 = {} and probe2 = {} to {}.'.format(i, ii, path2wsp))

if __name__ == '__main__':
    cls = PipelineStage.main()

