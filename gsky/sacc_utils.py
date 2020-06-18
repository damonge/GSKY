import logging
import numpy as np
import sacc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def coadd_saccs(saccfiles, tracers, ell_max_dict=None, trim_sacc=True, trc_combs=None):
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

        logger.info('Size of saccfile before trc and ell cuts {}.'.format(saccfile.mean.size))
        if trc_combs is not None:
            logger.info('trc_combs provided.')
            for tr_i, tr_j in saccfile.get_tracer_combinations():
                if (tr_i, tr_j) in trc_combs or (tr_j, tr_i) in trc_combs:
                    if ell_max_dict is not None:
                        ell_max_curr = min(ell_max_dict[tr_i], ell_max_dict[tr_j])
                        logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
                        saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
                else:
                    saccfile.remove_selection(tracers=(tr_i, tr_j))
        else:
            logger.info('trc_combs not provided.')
            for tr_i, tr_j in saccfile.get_tracer_combinations():
                if tr_i in tracers and tr_j in tracers:
                    if ell_max_dict is not None:
                        ell_max_curr = min(ell_max_dict[tr_i], ell_max_dict[tr_j])
                        logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
                        saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
                else:
                    saccfile.remove_selection(tracers=(tr_i, tr_j))
        logger.info('Size of saccfile after trc and ell cuts {}.'.format(saccfile.mean.size))

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

    assert set(tracers) <= set(tempsacc_tracers), 'Larger tracer set requested than present in largest ' \
                                                                 'saccfile. Aborting.'

    for i, saccfile in enumerate(sacc_coadds[1:]):
        sacc_tracers = saccfile.tracers.keys()
        missing_tracers = list(set(tracers) - set(sacc_tracers))
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

    if trim_sacc:
        logger.info('Trimming sacc - removing windows.')
        saccfile_coadd_trimmed = sacc.Sacc()
        for trc_name in saccfile_coadd.tracers.keys():
            saccfile_coadd_trimmed.add_tracer_object(saccfile_coadd.tracers[trc_name])
        datatypes = saccfile_coadd.get_data_types()
        for datatype in datatypes:
            tracer_combs = saccfile_coadd.get_tracer_combinations(data_type=datatype)
            for tr_i1, tr_j1 in tracer_combs:
                ell, cl = saccfile_coadd.get_ell_cl(datatype, tr_i1, tr_j1, return_cov=False)

                saccfile_coadd_trimmed.add_ell_cl(datatype, tr_i1, tr_j1, ell, cl)
        assert np.all(saccfile_coadd.mean == saccfile_coadd_trimmed.mean), 'Error while trimming sacc, means not equal. Aborting.'
        saccfile_coadd_trimmed.add_covariance(cov_coadd)
        saccfile_coadd = saccfile_coadd_trimmed
    else:
        logger.info('Not trimming sacc.')
        saccfile_coadd = coadd_sacc_windows(saccfiles, saccfile_coadd)

    return saccfile_coadd

def coadd_sacc_means(saccfiles, config):

    logger.info('Coadding means of saccfiles.')

    # for saccfile in saccfiles:
    #     logger.info('Initial size of saccfile = {}.'.format(saccfile.mean.size))
        # logger.info('Removing B-modes.')
        # saccfile.remove_selection(data_type='cl_eb')
        # saccfile.remove_selection(data_type='cl_be')
        # saccfile.remove_selection(data_type='cl_bb')
        # saccfile.remove_selection(data_type='cl_0b')
        # logger.info('Removing yxy.')
        # saccfile.remove_selection(data_type='cl_00', tracers=('y_0', 'y_0'))
        # logger.info('Removing kappaxkappa.')
        # saccfile.remove_selection(data_type='cl_00', tracers=('kappa_0', 'kappa_0'))
        # logger.info('Removing kappaxy.')
        # saccfile.remove_selection(data_type='cl_00', tracers=('kappa_0', 'y_0'))
        # saccfile.remove_selection(data_type='cl_00', tracers=('y_0', 'kappa_0'))
        # logger.info('Size of saccfile after cuts = {}.'.format(saccfile.mean.size))

        # logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
        # for tr_i, tr_j in saccfile.get_tracer_combinations():
        #     ell_max_curr = min(self.ell_max_dict[tr_i], self.ell_max_dict[tr_j])
        #     logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
        #     saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
        # logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

    for i, saccfile in enumerate(saccfiles):
        sacc_tracers = saccfile.tracers.keys()
        if set(sacc_tracers) == set(config['tracers']):
            tempsacc = saccfile
            ind_tmp = i
            logger.info('Found sacc with all requested tracers at {}.'.format(ind_tmp))
            break

    try:
        coadd_mean = tempsacc.mean
        datatypes = tempsacc.get_data_types()

        nmeans = np.ones_like(coadd_mean)

    except:
        raise RuntimeError('More tracers requested than contained in any of the provided sacc files. Aborting.')

    for i, saccfile in enumerate(saccfiles):
        if i != ind_tmp:
            sacc_tracers = saccfile.tracers.keys()
            if set(sacc_tracers).issubset(config['tracers']) and len(sacc_tracers) < len(config['tracers']):
                missing_tracers = list(set(config['tracers']) - set(sacc_tracers))
                logger.info('Found missing tracers {} in saccfile {}.'.format(missing_tracers, i))

            for datatype in datatypes:
                tracer_combs = tempsacc.get_tracer_combinations(data_type=datatype)
                for tr_i, tr_j in tracer_combs:
                    _, cl = saccfile.get_ell_cl(datatype, tr_i, tr_j, return_cov=False)

                    ind_here = saccfile.indices(data_type=datatype, tracers=(tr_i, tr_j))
                    ind_tempsacc = tempsacc.indices(data_type=datatype, tracers=(tr_i, tr_j))
                    if not ind_here.size == 0:
                        coadd_mean[ind_tempsacc] += cl
                        nmeans[ind_tempsacc] += 1

    coadd_mean /= nmeans

    # Copy sacc
    saccfile_coadd = tempsacc.copy()
    # Set mean of new saccfile to coadded mean
    saccfile_coadd.mean = coadd_mean
    saccfile_coadd.covariance = None

    return saccfile_coadd

def coadd_saccs_separate(saccfiles, tracers, ell_max_dict=None, weights=None, is_noisesacc=False):

        logger.info('Coadding saccfiles with common probes.')

        if weights is None:
            weights = np.ones(len(saccfiles))

        for i, saccfile in enumerate(saccfiles):
            logger.info('Removing B-modes.')
            saccfile.remove_selection(data_type='cl_eb')
            saccfile.remove_selection(data_type='cl_be')
            saccfile.remove_selection(data_type='cl_bb')
            saccfile.remove_selection(data_type='cl_0b')
            if not any('y_' in s for s in tracers) and not any('kappa_' in s for s in tracers):
                if any('y_' in key for key in saccfile.tracers.keys()):
                    for t in saccfile.tracers:
                        logger.info('Removing y_0.')
                        saccfile.remove_selection(tracers=('y_0', t))
                        saccfile.remove_selection(tracers=(t, 'y_0'))
                if any('kappa_' in key for key in saccfile.tracers.keys()):
                    for t in saccfile.tracers:
                        logger.info('Removing kappa_0.')
                        saccfile.remove_selection(tracers=('kappa_0', t))
                        saccfile.remove_selection(tracers=(t, 'kappa_0'))

                if ell_max_dict is not None:
                    logger.info('Size of saccfile before ell cuts {}.'.format(saccfile.mean.size))
                    for tr_i, tr_j in saccfile.get_tracer_combinations():
                        if tr_i in tracers and tr_j in tracers:
                            ell_max_curr = min(ell_max_dict[tr_i], ell_max_dict[tr_j])
                            logger.info('Removing ells > {} for {}, {}.'.format(ell_max_curr, tr_i, tr_j))
                            saccfile.remove_selection(tracers=(tr_i, tr_j), ell__gt=ell_max_curr)
                        else:
                            saccfile.remove_selection(tracers=(tr_i, tr_j))
                    logger.info('Size of saccfile after ell cuts {}.'.format(saccfile.mean.size))

            if i == 0:
                coadd_mean = weights[i]*saccfile.mean
                if not is_noisesacc:
                    coadd_cov = weights[i]**2*saccfile.covariance.covmat
            else:
                coadd_mean += weights[i]*saccfile.mean
                if not is_noisesacc:
                    coadd_cov += weights[i]**2*saccfile.covariance.covmat

        coadd_mean /= np.sum(weights)
        if not is_noisesacc:
            coadd_cov /= np.sum(weights) ** 2

        # Copy sacc
        saccfile_coadd = saccfiles[0].copy()
        # Set mean of new saccfile to coadded mean
        saccfile_coadd.mean = coadd_mean
        if not is_noisesacc:
            saccfile_coadd.add_covariance(coadd_cov)

        return saccfile_coadd

def coadd_sacc_windows(saccfiles, saccfile_coadd):

    logger.info('Coadding window functions.')

    # Add tracers to sacc
    tempsacc = sacc.Sacc()
    for trc_name, trc in saccfile_coadd.tracers.items():
        tempsacc.add_tracer_object(trc)

    datatypes = saccfile_coadd.get_data_types()
    trc_combs = saccfile_coadd.get_tracer_combinations()

    for tr_i, tr_j in trc_combs:
        for data_type in datatypes:
            ell_coadd_curr, cl_coadd_curr = saccfile_coadd.get_ell_cl(data_type, tr_i, tr_j, return_cov=False)
            # Get window
            if cl_coadd_curr != np.array([]):
                win_coadd = []
                n_wins = 0
                for sacc_curr in saccfiles:
                    # Query windows from indices
                    ind_curr = sacc_curr.indices(data_type=data_type, tracers=(tr_i, tr_j))
                    if ind_curr != []:
                        win_curr = sacc_curr.get_bandpower_windows(ind_curr)
                        if win_coadd != []:
                            win_coadd += win_curr.weight
                        else:
                            win_coadd = win_curr.weight
                            ell_coadd = win_curr.values
                        n_wins += 1

                logger.info('Subsampling windows with deltal = 14.')
                subsamp_winds_band = 14
                n_ell = win_coadd.shape[0]
                n_bands = win_coadd.shape[1]
                n_subsamp = n_ell // subsamp_winds_band

                win_coadd /= n_wins
                win_coadd_subsamp = win_coadd.reshape((n_bands, n_subsamp, subsamp_winds_band))
                win_coadd_subsamp = np.mean(win_coadd_subsamp, axis=-1)
                ell_subsamp = np.mean(ell_coadd.reshape(subsamp_winds_band, -1), axis=-1)

                win_coadd_subsamp = sacc.BandpowerWindow(ell_subsamp, win_coadd_subsamp.T)
                tempsacc.add_ell_cl(data_type, tr_i, tr_j, ell_coadd_curr, cl_coadd_curr, window=win_coadd_subsamp)

    return tempsacc