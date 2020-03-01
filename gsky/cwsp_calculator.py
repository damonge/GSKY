from ceci import PipelineStage
from .types import FitsFile,ASCIIFile,SACCFile,DummyFile
import numpy as np
import pymaster as nmt
from .cov_gauss import CovGauss

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO: Names of files to read
#TODO: COSMOS nz for shear weights

class CwspCalc(CovGauss) :
    name="CwspCalc"
    inputs=[('masked_fraction',FitsFile),('ngal_maps',FitsFile),('shear_maps',FitsFile),
            ('act_maps', FitsFile), ('dust_map',FitsFile),('star_map',FitsFile),
            ('depth_map',FitsFile),('ccdtemp_maps',FitsFile),('airmass_maps',FitsFile),
            ('exptime_maps',FitsFile),('skylevel_maps',FitsFile),('sigma_sky_maps',FitsFile),
            ('seeing_maps',FitsFile),('ellipt_maps',FitsFile),('nvisit_maps',FitsFile),
            ('cosmos_weights',FitsFile),('syst_masking_file',ASCIIFile)]
    outputs=[('dummy',DummyFile),
             ('cov_wodpj',SACCFile),('cov_wdpj',SACCFile)]
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
                    'output_run_dir': 'NONE','sys_collapse_type':'average',
                    'tracerCombInd': int}

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
        self.ncross = self.nmaps * (self.nmaps + 1) // 2 + self.ntracers_shear

        # Set up mapping
        self.mapping(tracers_nc)

        cwsp_curr = self.get_covar_mcm(tracers_wc, bpws, tracerCombInd=self.config['tracerCombInd'])

if __name__ == '__main__':
    cls = PipelineStage.main()
