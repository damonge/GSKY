from ceci import PipelineStage
from .types import FitsFile
from astropy.table import Table, hstack
import numpy as np
from .plot_utils import plot_histo
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COSMOSWeight(PipelineStage):
    name = "COSMOSWeight"
    inputs = [('cosmos_data', FitsFile),
              ('cosmos_hsc', FitsFile)]
    outputs = [('cosmos_weights', FitsFile)]
    config_options = {'depth_cut': 24.5,
                      'band': 'i',
                      'mask_type': 'sirius',
                      'n_neighbors': 10}

    def run(self):
        """
        Main function.
        This stage matches the COSMOS-30band data with the HSC COSMOS sample
        cut with the same criteria as our data and produces colour-space
        weights to match our sample so it can be used to estimate redshift
        distributions.
        """
        band = self.config['band']

        # Read HSC COSMOS catalog
        logger.info("Reading HSC COSMOS")
        cat = Table.read(self.get_input('cosmos_hsc'))
        # Clean nulls and nans
        sel = np.ones(len(cat), dtype=bool)
        isnull_names = []
        for key in cat.keys():
            if key.__contains__('isnull'):
                sel[cat[key]] = 0
                isnull_names.append(key)
            else:
                # Keep photo-z's even if they're NaNs
                if not key.startswith("pz_"):
                    sel[np.isnan(cat[key])] = 0
        cat.remove_columns(isnull_names)
        cat.remove_rows(~sel)
        # Collect sample cuts
        sel_maglim = np.ones(len(cat), dtype=bool)
        sel_maglim[cat['%scmodel_mag' % band] -
                   cat['a_%s' % band] > self.config['depth_cut']] = 0
        # Blending
        sel_blended = np.ones(len(cat), dtype=bool)
        # abs_flux<10^-0.375
        sel_blended[cat['iblendedness_abs_flux'] >= 0.42169650342] = 0
        # S/N in i
        sel_fluxcut_i = np.ones(len(cat), dtype=bool)
        sel_fluxcut_i[cat['icmodel_flux'] < 10*cat['icmodel_flux_err']] = 0
        # S/N in g
        sel_fluxcut_g = np.ones(len(cat), dtype=int)
        sel_fluxcut_g[cat['gcmodel_flux'] < 5*cat['gcmodel_flux_err']] = 0
        # S/N in r
        sel_fluxcut_r = np.ones(len(cat), dtype=int)
        sel_fluxcut_r[cat['rcmodel_flux'] < 5*cat['rcmodel_flux_err']] = 0
        # S/N in z
        sel_fluxcut_z = np.ones(len(cat), dtype=int)
        sel_fluxcut_z[cat['zcmodel_flux'] < 5*cat['zcmodel_flux_err']] = 0
        # S/N in y
        sel_fluxcut_y = np.ones(len(cat), dtype=int)
        sel_fluxcut_y[cat['ycmodel_flux'] < 5*cat['ycmodel_flux_err']] = 0
        # S/N in grzy (at least 2 pass)
        sel_fluxcut_grzy = (sel_fluxcut_g + sel_fluxcut_r +
                            sel_fluxcut_z + sel_fluxcut_y >= 2)
        # Overall S/N
        sel_fluxcut = sel_fluxcut_i*sel_fluxcut_grzy
        # Stars
        sel_stars = np.ones(len(cat), dtype=bool)
        sel_stars[cat['iclassification_extendedness'] > 0.99] = 0
        # Galaxies
        sel_gals = np.ones(len(cat), dtype=bool)
        sel_gals[cat['iclassification_extendedness'] < 0.99] = 0
        ####
        # Implement final cuts
        # - Mag. limit
        # - S/N cut
        # - Star-galaxy separator
        # - Blending
        sel = ~(sel_maglim * sel_gals * sel_fluxcut * sel_blended)
        cat.remove_rows(sel)

        ####
        # Read COSMOS-30band
        logger.info("Reading COSMOS 30band")
        cat30 = fits.open(self.get_input('cosmos_data'))[1].data
        lim_indices = np.where((0.01 < cat30['PHOTOZ']) &
                               (9 > cat30['PHOTOZ']) &
                               (cat30['TYPE'] == 0) &
                               (cat30['ZP_2'] < 0) &
                               (cat30['MASS_BEST'] > 7.5) &
                               (np.maximum(cat30['ZPDF_H68']-cat30['ZPDF'],
                                           cat30['ZPDF']-cat30['ZPDF_L68']) <
                                0.05*(1+cat30['PHOTOZ'])) &
                               (cat30['CHI2_BEST'] < cat30['CHIS']) &
                               (cat30['CHI2_BEST']/cat30['NBFILT'] < 5.))
        cat30 = cat30[lim_indices]

        ####
        # Match coordinates
        logger.info("Matching coordinates")
        cosmos_crd = SkyCoord(ra=np.array(cat30['ALPHA_J2000'])*u.deg,
                              dec=np.array(cat30['DELTA_J2000'])*u.deg)
        hsc_crd = SkyCoord(ra=np.array(cat['ra'])*u.deg,
                           dec=np.array(cat['dec'])*u.deg)
        # Nearest neighbors
        cosmos_index, dist_2d, dist_3d = hsc_crd.match_to_catalog_sky(cosmos_crd)
        # Cut everything further than 1 arcsec
        mask = dist_2d.degree*60*60 < 1
        cat_good = cat[mask]
        cat30_good = cat30[cosmos_index[mask]]
        cosmos_index_matched = cosmos_index[mask]

        t1 = Table.from_pandas(pd.DataFrame(cat30_good))
        keys_t2 = ['gcmodel_mag', 'rcmodel_mag', 'icmodel_mag', 'zcmodel_mag',
                   'ycmodel_mag', 'pz_mean_eab', 'pz_mode_eab', 'pz_best_eab',
                   'pz_mc_eab']
        t2 = Table.from_pandas(pd.DataFrame(np.transpose([np.array(cat_good[k])
                                                          for k in keys_t2]),
                                            index=range(len(cat_good)),
                                            columns=keys_t2))
        cat_matched = hstack([t1, t2])

        ####
        # Get color-space weights
        bands = ['g', 'r', 'i', 'z', 'y']
        logger.info("Computing color-space weights")
        train_sample = np.transpose(np.array([np.array(cat_matched['%scmodel_mag' % m])
                                              for m in bands]))
        photoz_sample = np.transpose(np.array([np.array(cat['%scmodel_mag' % m])
                                               for m in bands]))

        # Find nearest neighbors in color space
        n_nbrs = NearestNeighbors(n_neighbors=self.config['n_neighbors'],
                                  algorithm='kd_tree',
                                  metric='euclidean').fit(train_sample)
        distances, _ = n_nbrs.kneighbors(train_sample)
        # Get maximum distance
        distances = np.amax(distances, axis=1)
        # Find all photo-z objects within this maximum distance
        # for each COSMOS object
        tree_NN_lookup = spatial.cKDTree(photoz_sample, leafsize=40)
        num_photoz = np.array([len(tree_NN_lookup.query_ball_point(t, d+1E-6))
                               for t, d in zip(train_sample, distances)])
        # Weights are ratio of number of photo-z neighbors to
        # COSMOS neighbors (normalized by the number of photo-z objects)
        weights = np.true_divide(num_photoz*len(train_sample),
                                 self.config['n_neighbors'] *
                                 len(photoz_sample))
        logger.info(np.sum(weights))

        ####
        # Write output
        keys_t1 = ['ALPHA_J2000', 'DELTA_J2000', 'gcmodel_mag', 'rcmodel_mag',
                   'icmodel_mag', 'zcmodel_mag', 'ycmodel_mag',
                   'pz_best_eab', 'PHOTOZ', 'MNUV', 'MU', 'MB',
                   'MV', 'MR', 'MI', 'MZ', 'MY', 'MJ', 'MH', 'MK']
        t1 = Table.from_pandas(pd.DataFrame(np.transpose([cat_matched[k]
                                                          for k in keys_t1]),
                                            columns=keys_t1))
        t2 = Table.from_pandas(pd.DataFrame(np.transpose(weights),
                                            columns=['weight']))
        t3 = Table.from_pandas(pd.DataFrame(np.transpose(cosmos_index_matched),
                                            columns=['cosmos_index_matched']))
        cat_weights = hstack([t1, t2, t3])
        cat_weights.write(self.get_output('cosmos_weights'),
                          overwrite=True)

        # Plot magnitude distribution
        for b in ['g', 'r', 'i', 'z', 'y']:
            plot_histo(self.config, '%s_mag_COSMOS' % b,
                       [cat_weights['%scmodel_mag' % b],
                        cat['%scmodel_mag' % b]],
                       ['COSMOS', 'HSC'],
                       weights=[cat_weights['weight'],
                                np.ones(len(cat))],
                       density=True, logy=True, bins=50)
        # Plot weights distribution
        plot_histo(self.config, 'COSMOS_weights',
                   [cat_weights['weight']], ['weights'],
                   density=True, logy=True, bins=50)


if __name__ == '__main__':
    cls = PipelineStage.main()
