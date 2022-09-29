from ceci import PipelineStage
from .types import FitsFile, ASCIIFile, DummyFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createCountsMap
from astropy.io import fits
import os
from .plot_utils import plot_map, plot_curves
from gsky.noise_bias_from_mocks import NoiseBiasFromMocks

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoiseBiasMocks(PipelineStage):
    name = "NoiseBiasMocks"
    inputs = []
    outputs = [('dummy',DummyFile)]
    config_options={'plots_dir': None,
          'min_snr': 10., 'depth_cut': 24.5,
          'mapping': {'wcs': None, 'res': 0.0285,
                      'res_bo': 0.003, 'pad': 0.1,
                      'projection': 'CAR'},
          'band': 'i', 'depth_method': 'fluxerr',
          'shearrot': 'noflip', 'mask_type': 'sirius',
          'ra':  'i_ra', 'dec':  'i_dec',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shearrot': 'noflip',
          'ra':  'i_ra', 'dec':  'i_dec'}

    def get_output_fname(self,name,ext=None):
        self.output_dir=self.get_output('dummy',final_name=True)[:-5]
        if self.config['output_run_dir'] != 'NONE':
            self.output_dir+=self.config['output_run_dir']+'/'
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        fname=self.output_dir+name
        if ext is not None:
            fname+='.'+ext
        return fname

    def run(self):
        """
        Main routine. This stage:
        - Calls CovFromMocks in cov_from_mocks.py
        """
        test = NoiseBiasFromMocks()
        cls, ells = test.go()

        np.save(self.get_output_fname('noise_cls', 'npy'), cls)
        logger.info('Written signal cls to {}.'.format(self.get_output_fname('cls_signal_realiz_fourth_moment', ext='npy')))

        np.save(self.get_output_fname('l_eff_noise', 'npy'), ells)
        logger.info('Written ells to {}.'.format(self.get_output_fname('l_eff_noise_fourth_moment', ext='npy')))

if __name__ == '__main__':
    cls = PipelineStage.main()