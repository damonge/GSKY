import numpy as np
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def createCountsMap(ra, dec, fsk):
    """
    Creates a map containing the number of objects in each pixel.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param fsk: a flatmaps.FlatMapInfo object describing the
        geometry of the output map.
    """
    flatmap = fsk.pos2pix(ra, dec)
    mp = np.bincount(flatmap[flatmap >= 0],
                     weights=None, minlength=fsk.get_size())
    return mp


def createSpin2Map(ra, dec, q, u, fsk, weights=None, shearrot=None):
    """
    Creates two maps containing the averages (optionally weighted)
    of the Q, U components of a spin-2 field.
    :param ra:
    :param dec:
    :param q:
    :param u:
    :param fsk:
    :param weights:
    :param shearrot:
    :return:
    """

    flatmap = fsk.pos2pix(ra, dec)
    id_good = flatmap >= 0

    if weights is not None:
        q = weights*copy.deepcopy(q)
        u = weights*copy.deepcopy(u)

    qmap = np.bincount(flatmap[id_good],
                       weights=q[id_good],
                       minlength=fsk.get_size())
    umap = np.bincount(flatmap[id_good],
                       weights=u[id_good],
                       minlength=fsk.get_size())
    weightsmap = np.bincount(flatmap[id_good],
                             weights=weights,
                             minlength=fsk.get_size())
    nmap = np.bincount(flatmap[id_good],
                       weights=None,
                       minlength=fsk.get_size())

    qmap[weightsmap != 0] /= weightsmap[weightsmap != 0]
    umap[weightsmap != 0] /= weightsmap[weightsmap != 0]

    if weights is not None:
        logger.info('Weights provided.')
        logger.info('Computing weightmask.')
        weightmask = weightsmap
        logger.info('Computing binary mask.')
        mask = weightsmap != 0.
        mask = mask.astype('int')
    else:
        logger.info('No weights provided.')
        logger.info('Computing binary mask.')
        mask = weightsmap != 0.
        mask = mask.astype('int')
        weightmask = copy.deepcopy(mask)

    if shearrot is None:
        logger.info('shearrot is None. Not applying shear transformation.')

    else:
        if shearrot == 'flipqu':
            logger.info('shearrot is '
                        '{}. Applying shear transformation.'.format(shearrot))
            qmap *= (-1.)
            umap *= (-1.)
        elif shearrot == 'flipq':
            logger.info('shearrot is '
                        '{}. Applying shear transformation.'.format(shearrot))
            qmap *= (-1.)
        elif shearrot == 'flipu':
            logger.info('shearrot is '
                        '{}. Applying shear transformation.'.format(shearrot))
            umap *= (-1.)
        elif shearrot == 'noflip':
            logger.info('shearrot is '
                        '{}. Applying shear transformation.'.format(shearrot))
        else:
            logger.error('Accepted values of shearrot '
                         '= [noflip, flipq, flipu, flipqu].')

    mp = [qmap, umap]
    ms = [weightmask, mask, nmap]

    return mp, ms

def createWQUMap(ra, dec, q, u, fsk, weights=None):
    """
    Creates two maps containing the averages (optionally weighted)
    of the Q, U components of a spin-2 field.
    :param ra:
    :param dec:
    :param q:
    :param u:
    :param fsk:
    :param weights:
    :param shearrot:
    :return:
    """

    flatmap = fsk.pos2pix(ra, dec)
    id_good = flatmap >= 0

    if weights is None:
        weights = np.ones_like(q)

    wqmap = np.bincount(flatmap[id_good],
                       weights=q[id_good]*weights[id_good],
                       minlength=fsk.get_size())
    wumap = np.bincount(flatmap[id_good],
                       weights=u[id_good]*weights[id_good],
                       minlength=fsk.get_size())

    mp = [wqmap, wumap]

    return mp

def createW2QU2Map(ra, dec, q, u, fsk, weights=None):
    """
    Creates two maps containing the averages (optionally weighted)
    of the squares of Q, U components of a spin-2 field.
    :param ra:
    :param dec:
    :param q:
    :param u:
    :param fsk:
    :param weights:
    :param shearrot:
    :return:
    """

    flatmap = fsk.pos2pix(ra, dec)
    id_good = flatmap >= 0

    if weights is None:
        weights = np.ones_like(q)

    w2q2map = np.bincount(flatmap[id_good],
                       weights=q[id_good]**2*weights[id_good]**2,
                       minlength=fsk.get_size())
    w2u2map = np.bincount(flatmap[id_good],
                       weights=u[id_good]**2*weights[id_good]**2,
                       minlength=fsk.get_size())

    mp = [w2q2map, w2u2map]

    return mp

def createMeanStdMaps(ra, dec, quantity, fsk):
    """
    Creates maps of the mean and standard deviation of a given quantity
    measured at the position of a number of objects.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param quantity: measurements of the quantity to map for each object.
    :param fsk: a flatmaps.FlatMapInfo object describing the geometry of
        the output map.
    """
    pix_ids_old = fsk.pos2pix(ra, dec)
    pix_ids = np.ones(len(pix_ids_old))
    print("Len pix_ids", len(pix_ids))
    id_good = pix_ids >= 0
    print("id_good", id_good)
    print("Sum id_good", np.sum(id_good))
    mp = np.bincount(pix_ids[id_good],
                     weights=None,
                     minlength=fsk.get_size())
    mpW = np.bincount(pix_ids[id_good],
                      weights=quantity[id_good],
                      minlength=fsk.get_size())
    mpWSq = np.bincount(pix_ids[id_good],
                        weights=quantity[id_good]**2,
                        minlength=fsk.get_size())
    idgood = np.where(mp > 0)[0]
    test_idgood = np.ones(len(mp))
    print(len(mp))
    print("Sum test_idgood", np.sum(test_idgood[mp>0]))
    mean = np.zeros(len(mp))
    std = np.zeros(len(mp))
    mean[idgood] = mpW[idgood]/mp[idgood]
    std[idgood] = np.sqrt(np.fabs(((mpWSq[idgood]/mp[idgood]) -
                                   mean[idgood]**2)/(mp[idgood]+0.)))
    # mean = mpW/mp
    # std = np.sqrt(np.fabs(((mpWSq/mp) -
    #                                mean**2)/(mp+0.)))

    idbad = np.where(mp <= 0)[0]
    test_idbad = np.ones(len(mp))
    print("Sum test_idbad", np.sum(test_idbad[mp <= 0]))
    print(len(idbad))
    mean[idbad] = 0.0
    std[idbad] = 0

    return mean, std

def createSumMap(ra, dec, quantity, fsk):
    """
    Creates maps of the sum of a given quantity
    measured at the position of a number of objects.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param quantity: measurements of the quantity to map for each object.
    :param fsk: a flatmaps.FlatMapInfo object describing the geometry of
        the output map.
    """
    pix_ids = fsk.pos2pix(ra, dec)
    id_good = pix_ids >= 0
    print('pix_ids', pix_ids)
    mp = np.bincount(pix_ids[id_good],
                     weights=quantity[id_good],
                     minlength=fsk.get_size())

    return mp

def createMedianMap(ra, dec, quantity, fsk):
    """
    Creates maps of the sum of a given quantity
    measured at the position of a number of objects.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param quantity: measurements of the quantity to map for each object.
    :param fsk: a flatmaps.FlatMapInfo object describing the geometry of
        the output map.
    """
    pix_ids = fsk.pos2pix(ra, dec)
    id_good = pix_ids >= 0
    print('pix_ids', pix_ids)
    mp = np.zeros(fsk.get_size())
    for pix in np.unique(pix_ids[id_good]):
        mask = pix_ids[id_good]==pix
        mp[pix] = np.median(quantity[id_good][mask])

    return mp

def createMask(ra, dec, flags, flatsky_base, reso_mask):
    """
    Creates a mask based on the position of random objects and a set
    of flags.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param flags: list of arrays containing the flags used to mask
        areas of the sky. pixels containing objects with any
        flags=True will be masked. Pass [] if you just want to
        define a mask based on object positions.
    :param flatsky_base: FlatMapInfo for the base mask, defined by
        the presence of not of object in pixels defined by this
        FlatMapInfo
    :param reso_mask: resolution of the final mask (dx or dy)
    :return: mask and associated FlatMapInfo
    """
    from scipy.ndimage import label
    fsg0 = flatsky_base

    # Create mask based on object positions
    mpr = createCountsMap(ra, dec, fsg0)
    mskr = np.zeros(fsg0.get_size())
    mskr[mpr > 0] = 1

    if(np.sum(mpr*mskr)/np.sum(mskr) < 5):
        raise Warning('Base resolution may '
                      'be too high %.1lf' %
                      (np.sum(mpr*mskr)/np.sum(mskr)))

    if np.fabs(reso_mask) > np.fabs(fsg0.dx):
        fsg, mpn = fsg0.d_grade(mpr,
                                int(np.fabs(reso_mask/fsg0.dx)+0.5))
    else:
        fsg, mpn = fsg0.u_grade(mpr,
                                int(np.fabs(fsg0.dx/reso_mask)+0.5))

    mskn = np.zeros(fsg.get_size())
    mskn[mpn > 0] = 1
    ipix = fsg.pos2pix(ra, dec)
    for flag in flags:
        ipixmask = ipix[flag]
        p = np.unique(ipixmask)
        mskn[p] = 0

    # Classify all connected regions
    msk2d = mskn.astype(int).reshape([fsg.ny, fsg.nx])
    labeled_array, num_features = label(msk2d)
    # Identify largest connected region (0 is background)
    i0 = np.argmax(np.histogram(labeled_array,
                                bins=num_features+1,
                                range=[-0.5, num_features+0.5])[0][1:])+1
    # Remove all other regions
    mask_clean = labeled_array.copy().flatten()
    mask_clean[mask_clean != i0] = 0
    msk_out = mask_clean.astype(float)

    return msk_out, fsg


def removeDisconnected(mp, fsk):
    from scipy.ndimage import label
    labeled_array, num_features = label(mp.reshape([fsk.ny, fsk.nx]))
    i0 = np.argmax(np.histogram(labeled_array,
                                bins=num_features+1,
                                range=[-0.5, num_features+0.5])[0][1:])+1
    mask_clean = labeled_array.copy().flatten()
    mpo = mp.copy()
    mpo[mask_clean != i0] = 0

    return mpo
