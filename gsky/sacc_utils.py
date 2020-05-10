import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def coadd_saccs(saccfiles):
    logger.info('Coadding all saccfiles weighted by inverse variance.')

    for saccfile in saccfiles:
        logger.info('Initial size of saccfile = {}.'.format(saccfile.mean.size))
        logger.info('Removing B-modes.')
        saccfile.remove_selection(data_type='cl_eb')
        saccfile.remove_selection(data_type='cl_be')
        saccfile.remove_selection(data_type='cl_bb')
        saccfile.remove_selection(data_type='cl_0b')
        logger.info('Removing yxy.')
        saccfile.remove_selection(data_type='cl_00', tracers=('y_0', 'y_0'))
        logger.info('Removing kappaxkappa.')
        saccfile.remove_selection(data_type='cl_00', tracers=('kappa_0', 'kappa_0'))
        logger.info('Removing kappaxy.')
        saccfile.remove_selection(data_type='cl_00', tracers=('kappa_0', 'y_0'))
        saccfile.remove_selection(data_type='cl_00', tracers=('y_0', 'kappa_0'))
        logger.info('Size of saccfile after cuts = {}.'.format(saccfile.mean.size))

        logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
        for tr_i, tr_j in saccfile.get_tracer_combinations():
            ell_max_curr = min(self.ell_max_dict[tr_i], self.ell_max_dict[tr_j])
            logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
            saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
        logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

    ntracers_arr = np.array([len(saccfile.tracers) for saccfile in saccfiles])
    ntracers_unique = np.unique(ntracers_arr)[::-1]

    saccs_list = [[] for i in range(ntracers_unique.shape[0])]
    for i in range(ntracers_unique.shape[0]):
        for saccfile in saccfiles:
            if len(saccfile.tracers) == ntracers_unique[i]:
                saccs_list[i].append(saccfile)

    sacc_coadds = [0 for i in range(ntracers_unique.shape[0])]
    for i in range(ntracers_unique.shape[0]):
        len_curr = ntracers_unique[i]
        nsacc_curr = len(saccs_list[i])
        logger.info('Found {} saccfiles of length {}.'.format(nsacc_curr, len_curr))
        for j, saccfile in enumerate(saccs_list[i]):
            if j == 0:
                coadd_mean = saccfile.mean
                coadd_cov = saccfile.covariance.covmat
            else:
                coadd_mean += saccfile.mean
                coadd_cov += saccfile.covariance.covmat

        coadd_mean /= nsacc_curr
        coadd_cov /= nsacc_curr ** 2

        # Copy sacc
        saccfile_coadd = saccfile.copy()
        # Set mean of new saccfile to coadded mean
        saccfile_coadd.mean = coadd_mean
        saccfile_coadd.add_covariance(coadd_cov)
        sacc_coadds[i] = saccfile_coadd

    tempsacc = sacc_coadds[0]
    tempsacc_tracers = tempsacc.tracers.keys()
    datatypes = tempsacc.get_data_types()
    invcov_coadd = np.linalg.inv(tempsacc.covariance.covmat)
    mean_coadd = np.dot(invcov_coadd, tempsacc.mean)

    assert set(tempsacc_tracers) == set(self.config['tracers']), 'Different tracers requested than present in largest ' \
                                                                 'saccfile. Aborting.'

    for i, saccfile in enumerate(sacc_coadds[1:]):
        sacc_tracers = saccfile.tracers.keys()
        missing_tracers = list(set(self.config['tracers']) - set(sacc_tracers))
        logger.info('Found missing tracers {} in saccfile {}.'.format(missing_tracers, i))

        invcov_small_curr = np.linalg.inv(saccfile.covariance.covmat)

        mean_big_curr = np.zeros_like(tempsacc.mean)
        invcov_big_curr = np.zeros_like(tempsacc.covariance.covmat)

        for datatype in datatypes:
            tracer_combs = tempsacc.get_tracer_combinations(data_type=datatype)
            for tr_i1, tr_j1 in tracer_combs:
                _, cl = saccfile.get_ell_cl(datatype, tr_i1, tr_j1, return_cov=False)

                ind_here = saccfile.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                ind_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                if not ind_here.size == 0:
                    mean_big_curr[ind_tempsacc] = cl
                for tr_i2, tr_j2 in tracer_combs:
                    ind_i1j1_curr = saccfile.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                    ind_i2j2_curr = saccfile.indices(data_type=datatype, tracers=(tr_i2, tr_j2))

                    subinvcov_curr = invcov_small_curr[np.ix_(ind_i1j1_curr, ind_i2j2_curr)]

                    ind_i1j1_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i1, tr_j1))
                    ind_i2j2_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i2, tr_j2))

                    if ind_i1j1_curr.size != 0 and ind_i2j2_curr.size != 0:
                        invcov_big_curr[np.ix_(ind_i1j1_tempsacc, ind_i2j2_tempsacc)] = subinvcov_curr

        mean_coadd += np.dot(invcov_big_curr, mean_big_curr)
        invcov_coadd += invcov_big_curr

    # Copy sacc
    saccfile_coadd = tempsacc.copy()
    # Set mean of new saccfile to coadded mean
    cov_coadd = np.linalg.inv(invcov_coadd)
    saccfile_coadd.mean = np.dot(cov_coadd, mean_coadd)
    saccfile_coadd.add_covariance(cov_coadd)

    return saccfile_coadd