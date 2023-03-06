from ceci import PipelineStage
from .types import FitsFile
import numpy as np
from .flatmaps import read_flat_map
from astropy.io import fits
import os
from .plot_utils import plot_map

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACTMapper(PipelineStage):
    name = "ACTMapper"
    inputs = [('masked_fraction', FitsFile)]
    outputs = [('act_maps', FitsFile)]
    config_options = {'act_inputs': ['none']}

    def check_fsks(self, fsk1, fsk2):
        """ Compares two flat-sky pixelizations
        """
        if((fsk1.nx == fsk2.nx) and (fsk1.ny == fsk2.ny) and
           (fsk1.dx == fsk2.dx) and (fsk1.dy == fsk2.dy)):
            return False
        return True

    def check_sanity(self, pix):
        """ Given the pixel indices of the corners of the HSC
        footprint in the ACT wcs, make sure that they make
        sense.
        """
        # Only two edges in x
        ix_unique = np.unique(pix[:, 0])
        check_a = len(ix_unique) == 2

        # Only two edges in y
        iy_unique = np.unique(pix[:, 1])
        check_b = len(iy_unique) == 2

        # Right separation between edges
        nx = int(np.fabs(np.diff(ix_unique)))
        ny = int(np.fabs(np.diff(iy_unique)))
        check_c = (nx == self.fsk_hsc.nx) and (ny == self.fsk_hsc.ny)

        # Integer pixel coordinates
        check_d = np.all(np.fabs(iy_unique - np.rint(iy_unique)) < 1E-5)
        check_e = np.all(np.fabs(ix_unique - np.rint(ix_unique)) < 1E-5)

        if not (check_a * check_b * check_c * check_d * check_e):
            raise ValueError("Sanity checks don't pass")

    def compute_edges(self):
        """
        Compute edges of the ACT map within HSC footprint
        """
        fsk = self.fsk_hsc
        self.coords_corner = fsk.wcs.all_pix2world([[0, 0],
                                                    [fsk.nx, 0],
                                                    [0, fsk.ny],
                                                    [fsk.nx, fsk.ny]],
                                                   0)
        pix = self.fsk_act.wcs.all_world2pix(self.coords_corner, 0)
        self.check_sanity(pix)

        self.ix0_act = int(np.amin(np.unique(pix[:, 0])))
        self.ixf_act = int(np.amax(np.unique(pix[:, 0])))
        self.iy0_act = int(np.amin(np.unique(pix[:, 1])))
        self.iyf_act = int(np.amax(np.unique(pix[:, 1])))

        # Translate in case HSC lies partially outside of ACT
        self.iyf_hsc = self.fsk_hsc.ny
        if self.iy0_act < 0:
            self.iyf_hsc += self.iy0_act
            self.iy0_act = 0

        self.iy0_hsc = 0
        if self.iyf_act > self.fsk_act.ny:
            self.iy0_hsc += self.iyf_act - self.fsk_act.ny
            self.iyf_act = self.fsk_act.ny

        self.ixf_hsc = self.fsk_hsc.nx
        if self.ix0_act < 0:
            self.ixf_hsc += self.ix0_act
            self.ix0_act = 0

        self.ix0_hsc = 0
        if self.ixf_act > self.fsk_act.nx:
            self.ix0_hsc += self.ixf_act - self.fsk_act.nx
            self.ixf_act = self.fsk_act.nx

    def read_maps(self):
        """ Reads sky geometry for HSC and ACT,
        as well as all the ACT maps and masks.
        """
        # HSC
        self.fsk_hsc, _ = read_flat_map(self.get_input("masked_fraction"))

        # ACT maps
        self.act_maps_full = []
        self.fsk_act = None
        if ((len(self.config['act_inputs']) == 1) and
            (self.config['act_inputs'][0] == 'none')):
            return

        for d in self.config['act_inputs']:
            mdir = {}
            fskb, msk = read_flat_map(d[2])
            fskc, mpp = read_flat_map(d[1])
            if self.check_fsks(fskb, fskc):
                raise ValueError("Footprints are incompatible")
            if self.fsk_act is None:
                self.fsk_act = fskb
            else:
                if self.check_fsks(fskb, self.fsk_act):
                    raise ValueError("ACT footprints are inconsistent")
            mdir['name'] = d[0]
            mdir['mask'] = msk.reshape([self.fsk_act.ny, self.fsk_act.nx])
            mdir['map'] = mpp.reshape([self.fsk_act.ny, self.fsk_act.nx])
            self.act_maps_full.append(mdir)

    def cut_act_map(self, mp):
        """ Returns an input ACT map cut to the HSC footprint.
        """
        mp_out = np.zeros([self.fsk_hsc.ny, self.fsk_hsc.nx])
        print('Shape of zeros map is', mp_out.shape)
        print('HSC initial and final y is', self.iy0_hsc, self.iyf_hsc)
        print('HSC initial and final x is', self.ix0_hsc, self.ixf_hsc)
        print('ACT initial and final y is', self.iy0_act, self.iyf_act)
        print('ACT initial and final x is', self.ix0_act, self.ixf_act)
        mp_out[self.iy0_hsc:self.iyf_hsc,
               self.ix0_hsc:self.ixf_hsc] = mp[self.iy0_act:self.iyf_act,
                                               self.ix0_act:self.ixf_act]
        return mp_out # subtract 1 from y dimension on right?

    def run(self):
        """
        Main routine. This stage:
        - Creates number density maps from the reduced catalog
          for a set of redshift bins.
        - Calculates the associated N(z)s for each bin using different
          methods.
        - Stores the above into a single FITS file
        """
        logger.info("Reading maps")
        self.read_maps()

        if self.fsk_act is not None:
            logger.info("Computing cutting edges")
            self.compute_edges()

            logger.info("Cutting maps")
            self.act_maps_hsc = []
            for d in self.act_maps_full:
                logger.info(" - " + d['name'])
                mpp = self.cut_act_map(d['map'])
                msk = self.cut_act_map(d['mask'])
                mdir = {}
                mdir['name'] = d['name']
                mdir['mask'] = msk
                mdir['map'] = mpp
                self.act_maps_hsc.append(mdir)

        logger.info("Writing output")
        header = self.fsk_hsc.wcs.to_header()
        hdus = []
        if self.fsk_act is None:
            head = header.copy()
            hdu = fits.PrimaryHDU(header=head)
            hdus.append(hdu)
        else:
            for im, d in enumerate(self.act_maps_hsc):
                head = header.copy()
                head['DESCR'] = d['name']
                if im == 0:
                    hdu = fits.PrimaryHDU(data=d['map'], header=head)
                else:
                    hdu = fits.ImageHDU(data=d['map'], header=head)
                hdus.append(hdu)
                head = header.copy()
                head['DESCR'] = d['name'] + ' mask'
                hdu = fits.ImageHDU(data=d['mask'], header=head)
                hdus.append(hdu)
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.get_output('act_maps'), overwrite=True)

        # Plotting
        if self.fsk_act is not None:
            for im, d in enumerate(self.act_maps_hsc):
                plot_map(self.config, self.fsk_hsc, d['map'].flatten(),
                         'act_' + d['name'])
                plot_map(self.config, self.fsk_hsc, d['mask'].flatten(),
                         'act_mask_' + d['name'])

        # Permissions on NERSC
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')


if __name__ == '__main__':
    cls = PipelineStage.main()
