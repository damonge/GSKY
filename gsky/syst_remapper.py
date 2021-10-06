from ceci import PipelineStage
from .types import FitsFile, HspFile
import numpy as np
from .flatmaps import read_flat_map
import os
from .map_utils import createMeanStdMaps, createSumMap
import healsparse as hsp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystReMapper(PipelineStage) :
    name="SystReMapper"
    # inputs=[('ccdtemp_maps',HspFile),('airmass_maps',HspFile),('exptime_maps',HspFile),
    #          ('skylevel_maps',HspFile),('sigma_sky_maps',HspFile),('ellipt_maps',HspFile),
    #          ('nvisit_maps',HspFile),('masked_fraction',FitsFile)]
    # outputs=[('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),('exptime_maps',FitsFile),
    #          ('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),('ellipt_maps',FitsFile),
    #          ('nvisit_maps',FitsFile)]

    inputs=[('airmass_maps',HspFile),('exptime_maps',HspFile),
             ('skylevel_maps',HspFile),('sigma_sky_maps',HspFile),('e1_maps',HspFile),
             ('e2_maps',HspFile),('nexp_maps',HspFile),('masked_fraction',FitsFile)]
    outputs=[('airmass_maps_out',FitsFile),('exptime_maps_out',FitsFile),
             ('skylevel_maps_out',FitsFile),('sigma_sky_maps_out',FitsFile),('e1_maps_out',FitsFile),
             ('e2_maps_out',FitsFile),('nexp_maps_out',FitsFile)]


    def run(self) :
        quants=['airmass','exptime','skylevel','sigma_sky','e1', 'e2', 'nexp']
        bands=['i']

        logger.info("Reading sample map")
        fsk,mp=read_flat_map(self.get_input('masked_fraction'))

        logger.info("Computing systematics maps")
        #Initialize maps
        oc_mean_maps = {}
        oc_std_maps = {}
        oc_med_maps = {}
        oc_sum_maps = {}
        for q in quants:
            if q != 'nexp' and q != 'exptime':
                oc_mean_maps[q] = {}
                oc_std_maps[q] = {}
                oc_med_maps[q] = {}
                for b in bands:
                    # TODO: Figure out naming
                    hsp_map = hsp.HealSparseMap.read(self.get_input(q+'_maps'))
                    vals = hsp_map[hsp_map.valid_pixels]
                    ra, dec = hsp_map.valid_pixels_pos(lonlat=True)
                    mean_map, std_map = createMeanStdMaps(ra, dec, vals, fsk)
                    median_map = createSumMap(ra, dec, vals, fsk)
                    oc_mean_maps[q][b] = mean_map
                    oc_std_maps[q][b] = std_map
                    oc_med_maps[q][b] = median_map
            else:
                oc_sum_maps[q] = {}
                for b in bands:
                    # TODO: Figure out naming
                    hsp_map = hsp.HealSparseMap.read(self.get_input(q+'_maps'))
                    vals = hsp_map[hsp_map.valid_pixels]
                    ra, dec = hsp_map.valid_pixels_pos(lonlat=True)
                    sum_map = createSumMap(ra, dec, vals, fsk)
                    oc_sum_maps[q][b] = sum_map

        logger.info("Saving maps")
        for q in quants:
            if q != 'nexp' and q != 'exptime':
                # Observing conditions
                maps_save = np.array([oc_mean_maps[q][b] for b in bands] +
                                   [oc_std_maps[q][b] for b in bands] +
                                   [oc_med_maps[q][b] for b in bands])
                descripts = np.array(['mean '+q+'-'+b for b in bands] +
                                   ['std '+q+'-'+b for b in bands] +
                                   ['median '+q+'-'+b for b in bands])
                fsk.write_flat_map(self.get_output(q+'_maps_out'),maps_save,descripts)
            else:
                # Nvisits
                maps_save = np.array([oc_sum_maps[q][b] for b in bands])
                descripts = np.array(['nvisit'+b for b in bands])
                fsk.write_flat_map(self.get_output(q+'_maps_out'), maps_save, descripts)

        # Permissions on NERSC
        # os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        # os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

if __name__ == '__main__':
    cls = PipelineStage.main()
