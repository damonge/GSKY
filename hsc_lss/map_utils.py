import numpy as np
import copy
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def createCountsMap(ra, dec, flatSkyGrid):
    """
    Creates a map containing the number of objects in each pixel.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param flatSkyGrid: a flatmaps.FlatMapInfo object describing the geometry of the output map.
    """
    flatmap= flatSkyGrid.pos2pix(ra, dec)
    mp= np.bincount(flatmap, weights= None, minlength= flatSkyGrid.get_size())
    
    return mp

def createSpin2Map(ra, dec, q, u, flatSkyGrid, weights=None, shearrot=None):
    """
    Creates two maps containing the averages (optionally weighted) of the Q, U components of a
    spin-2 field.
    :param ra:
    :param dec:
    :param q:
    :param u:
    :param flatSkyGrid:
    :param weights:
    :param shearrot:
    :return:
    """

    flatmap = flatSkyGrid.pos2pix(ra, dec)

    if weights is not None:
        q = weights*copy.deepcopy(q)
        u = weights*copy.deepcopy(u)

    qmap = np.bincount(flatmap, weights=q, minlength=flatSkyGrid.get_size())
    umap = np.bincount(flatmap, weights=u, minlength=flatSkyGrid.get_size())
    weightsmap = np.bincount(flatmap, weights=weights, minlength=flatSkyGrid.get_size())
    nmap = np.bincount(flatmap, weights=None, minlength=flatSkyGrid.get_size())

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
            logger.info('shearrot is {}. Applying shear transformation.'.format(shearrot))
            qmap *= (-1.)
            umap *= (-1.)
        elif shearrot == 'flipq':
            logger.info('shearrot is {}. Applying shear transformation.'.format(shearrot))
            qmap *= (-1.)
        elif shearrot == 'flipu':
            logger.info('shearrot is {}. Applying shear transformation.'.format(shearrot))
            umap *= (-1.)
        elif shearrot == 'noflip':
            logger.info('shearrot is {}. Applying shear transformation.'.format(shearrot))
        else:
            logger.error('Accepted values of shearrot = [noflip, flipq, flipu, flipqu].')

    mp = [qmap, umap]
    ms = [weightmask, mask, nmap]

    return mp, ms

def createMeanStdMaps(ra, dec, quantity, flatSkyGrid) :
    """
    Creates maps of the mean and standard deviation of a given quantity measured at the position 
    of a number of objects.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param quantity: measurements of the quantity to map for each object.
    :param flatSkyGrid: a flatmaps.FlatMapInfo object describing the geometry of the output map.
    """
    pix_ids=flatSkyGrid.pos2pix(ra, dec)
    mp= np.bincount(pix_ids, weights= None, minlength= flatSkyGrid.get_size())
    mpWeighted= np.bincount(pix_ids, weights= quantity, minlength= flatSkyGrid.get_size())
    mpWeightedSq= np.bincount(pix_ids, weights= quantity**2, minlength= flatSkyGrid.get_size())
    idgood=np.where(mp>0)[0];
    mean=np.zeros(len(mp)); std=np.zeros(len(mp))
    mean[idgood]= mpWeighted[idgood]/mp[idgood]
    std[idgood]= np.sqrt(np.fabs(((mpWeightedSq[idgood]/mp[idgood])-mean[idgood]**2)/(mp[idgood]+0.)))
    idbad=np.where(mp<=0)[0]
    mean[idbad]=0
    std[idbad]=0

    return mean, std

def createMask(ra,dec,flags,flatsky_base,reso_mask) :
    """
    Creates a mask based on the position of random objects and a set of flags.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param flags: list of arrays containing the flags used to mask areas of the sky.
                  pixels containing objects with any flags=True will be masked.
                  Pass [] if you just want to define a mask based on object positions.
    :param flatsky_base: FlatMapInfo for the base mask, defined by the presence of not
                   of object in pixels defined by this FlatMapInfo
    :param reso_mask: resolution of the final mask (dx or dy)
    :return: mask and associated FlatMapInfo
    """
    from scipy.ndimage import label
    fsg0=flatsky_base

    #Create mask based on object positions
    mpr=createCountsMap(ra,dec,fsg0)
    mskr=np.zeros(fsg0.get_size()); mskr[mpr>0]=1

    if(np.sum(mpr*mskr)/np.sum(mskr)<5) :
        raise Warning('Base resolution may be too high %.1lf'%(np.sum(mpr*mskr)/np.sum(mskr)))

    if np.fabs(reso_mask)>np.fabs(fsg0.dx) :
        fsg,mpn=fsg0.d_grade(mpr,int(np.fabs(reso_mask/fsg0.dx)+0.5))
    else :
        fsg,mpn=fsg0.u_grade(mpr,int(np.fabs(fsg0.dx/reso_mask)+0.5))

    mskn=np.zeros(fsg.get_size()); mskn[mpn>0]=1
    ipix=fsg.pos2pix(ra,dec)
    for flag in flags :
        ipixmask=ipix[flag]
        p=np.unique(ipixmask)
        mskn[p]=0

    #Classify all connected regions
    msk2d=mskn.astype(int).reshape([fsg.ny,fsg.nx])
    labeled_array,num_features=label(msk2d)
    #Identify largest connected region (0 is background)
    i0=np.argmax(np.histogram(labeled_array,bins=num_features+1,range=[-0.5,num_features+0.5])[0][1:])+1
    #Remove all other regions
    mask_clean=labeled_array.copy().flatten()
    mask_clean[mask_clean!=i0]=0
    msk_out=mask_clean.astype(float)
    
    return msk_out,fsg

def removeDisconnected(mp,fsk) :
    from scipy.ndimage import label
    labeled_array,num_features=label(mp.reshape([fsk.ny,fsk.nx]))
    i0=np.argmax(np.histogram(labeled_array,bins=num_features+1,range=[-0.5,num_features+0.5])[0][1:])+1
    mask_clean=labeled_array.copy().flatten()
    mpo=mp.copy()
    mpo[mask_clean!=i0]=0

    return mpo
