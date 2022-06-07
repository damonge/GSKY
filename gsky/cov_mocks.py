from ceci import PipelineStage
from .types import FitsFile, ASCIIFile
import numpy as np
from .flatmaps import read_flat_map
from .map_utils import createCountsMap
from astropy.io import fits
import os
from .plot_utils import plot_map, plot_curves
from gsky.cov_from_mocks import CovFromMocks

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CovMocks(PipelineStage):
    name = "CovMocks"
    inputs = []
    outputs = []
    config_options={'plots_dir': None,
          'min_snr': 10., 'depth_cut': 24.5,
          'mapping': {'wcs': None, 'res': 0.0285,
                      'res_bo': 0.003, 'pad': 0.1,
                      'projection': 'CAR'},
          'band': 'i', 'depth_method': 'fluxerr',
          'shearrot': 'noflip', 'mask_type': 'sirius',
          'ra':  'ra_mock', 'dec':  'dec_mock',
          'pz_code': 'dnnz', 'pz_mark': 'best',
          'pz_bins': [0.3, 0.6, 0.9, 1.2, 1.5],
          'nz_bin_num': 100,
          'nz_bin_max': 4.0,
          'shearrot': 'noflip',
          'ra':  'ra_mock', 'dec':  'dec_mock', 'shape_noise': True,
          'mocks_dir': '/projects/HSC/weaklens/xlshare/S19ACatalogs/catalog_mock/fields/XMM/'}

    def run(self):
        """
        Main routine. This stage:
        - Calls CovFromMocks in cov_from_mocks.py
        """
        test = CovFromMocks()
        test.go()

if __name__ == '__main__':
    cls = PipelineStage.main()