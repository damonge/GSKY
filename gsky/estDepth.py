import numpy as np
from .map_utils import createMeanStdMaps, createCountsMap

#############################################
# code from Javier: /global/projecta/projectdirs/lsst/groups/LSS/DC1/scripts/map_utils.py
# creates 5sigma depth HEALPix maps
def binned_statistic(x, values, func, nbins, range):
    '''The usage is approximately the same as the scipy one
    from https://stackoverflow.com/questions/26783719/effic
    iently-get-indices-of-histogram-bins-in-python'''
    from scipy.sparse import csr_matrix
    r0, r1 = range
    mask = (x > r0) &  (x < r1)
    x = x[mask]
    values = values[mask]
    N = len(values)
    digitized = (float(nbins) / (r1-r0) * (x-r0)).astype(int)
    S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))
    return np.array([func(group) for group in np.split(S.data, S.indptr[1:-1])])

def depth_map_snr_nonHP(ra, dec, mags, snr, snrthreshold, flatSkyGrid):
    # not based on healpix, original version modified to use flatmaps
    # also added the functionality to add snr_threshold
    good = np.logical_or(np.logical_not(np.isnan(ra)),np.logical_not(np.isnan(dec)))
    pix_nums = np.array(flatSkyGrid.pos2pix(ra, dec))
    
    map_out = np.zeros(flatSkyGrid.get_size())
    map_var_out = np.zeros(flatSkyGrid.get_size())
    mask_nans=np.zeros(flatSkyGrid.get_size());

    #Binned statistic 2d is awfully slow (because it doesn't use the fact that all bins are equal width
    #median_snr, xed, _, _ = binned_statistic_2d(mags,pix_nums,snr,statistic='median',bins=(50,12*nside**2),
    #                                           range=[(20,30),(0,12*nside**2)])
    #bin_centers = 0.5*xed[1:]+0.5*xed[:-1]
    #depth = bin_centers[np.argmin(np.fabs(median_snr-5),axis=0)]
    
    bin_centers = np.linspace(22+6/30.,28-6/30.,30.)
    for px in np.unique(pix_nums):
        mask_nans[px]=1
        mask = px==pix_nums
        if np.count_nonzero(mask)>0:
            median_snr = binned_statistic(mags[mask],snr[mask],np.nanmedian, nbins=30, range=(22,28))
            std_snr = binned_statistic(mags[mask],snr[mask],np.nanstd, nbins=30, range=(22,28))
            
            mask2 = np.isnan(median_snr)==False
            if np.count_nonzero(mask2)>0:
                depth = bin_centers[mask2][np.argmin(np.fabs(median_snr[mask2]-snrthreshold))]
                std = std_snr[np.argmin(np.fabs(median_snr[mask2]-snrthreshold))]
                map_out[px]=depth
                map_var_out[px]= std
            else:
                map_out[px]=0
                map_var_out[px]= 0
        else:
            map_out[px]=0.
            map_var_out[px]= 0

    map_out[mask_nans<1]=0
    map_var_out[mask_nans<1]=0

    return map_out, map_var_out

def desc_method(ra, dec, band, mags, snr, flatSkyGrid, SNRthreshold= 5):
    # make a histograms of the S/N in bins of magnitude for all objects in a given pixel
    # define the 5 sigma depth as the magnitude of the histogram whose median S/N is ~5.
    # SNRthreshold= 5 => 5sigma depth. can tweak it.
    
    depth, depth_std= depth_map_snr_nonHP(ra, dec,
                                          mags= mags,
                                          snr= snr,
                                          snrthreshold= SNRthreshold,
                                          flatSkyGrid= flatSkyGrid)    

    return depth, depth_std

#############################################

def flux_err_method(ra, dec, band, flux_err, flatSkyGrid, SNRthreshold= 5):
    # 5sigma Magnitude limit= average of 5*flux_err for all objs in each pixel (and then transformed to magnitude)
    # SNRthreshold= 5 => 5sigma depth.
    
    # Since want mags (mean, std) at the end, need to first run the createMeanStdMaps with 
    # 5flux_error to get the mean flux map which can be converted to fluxea.
    # To get std mags, need to run createMeanStdMaps with 5*flux_error converted to mags and keep only the std.

    depth, dontcare = createMeanStdMaps(ra, dec,
                                        quantity= SNRthreshold*flux_err,
                                        flatSkyGrid= flatSkyGrid)
    
    # convert from fluxes to mags
    depth= 10.**(23+6)*depth
    depth[~np.isnan(depth)]= -2.5*np.log10(depth[~np.isnan(depth)])+23.9
    
    # find the std.
    quantity= 10.**(23+6)*SNRthreshold*flux_err
    quanity= -2.5*np.log10(quantity)+23.9
    dontcare, depth_std= createMeanStdMaps(ra, dec,
                                           quantity= quantity,
                                           flatSkyGrid= flatSkyGrid)

    #Zeros in empty pixels
    nc=createCountsMap(ra,dec,flatSkyGrid)
    depth[nc<1]=0
    depth_std[nc<1]=0

    return depth, depth_std

#############################################
def random_sky_std_method(ra, dec, sky_std, band, flatSkyGrid, SNRthreshold= 5, outputDir= None):
    # 5sigma depth= -2.5*np.log10(SNRthreshold*sky_std)+27.0,
    # Using the random sky_std as here: https://hsc-release.mtk.nao.ac.jp/doc/index.php/random-points-for-dr1/
    # SNRthreshold= 5 => 5sigma depth.
    SNRthreshold= int(SNRthreshold)
    sky_std= np.array(sky_std)
    depth, depth_std = createMeanStdMaps(ra, dec,
                                         quantity= -2.5*np.log10(SNRthreshold*sky_std)+27.0,
                                         flatSkyGrid= flatSkyGrid)
        
    return depth, depth_std

#############################################
def depth_map_meanSNRrange(ra, dec, mags, snr, snrthreshold, flatSkyGrid):
    # 5sigma depth= mean of mags of galaxies with 4<SNR<6
    good = np.logical_or(np.logical_not(np.isnan(ra)),np.logical_not(np.isnan(dec)))
    pix_nums = np.array(flatSkyGrid.pos2pix(ra, dec))

    map_out = np.zeros(flatSkyGrid.get_size());
    map_var_out = np.zeros(flatSkyGrid.get_size());
    mask_nans=np.zeros(flatSkyGrid.get_size());
    
    for px in np.unique(pix_nums):
        mask_nans[px]=1
        mask = px==pix_nums
        if np.count_nonzero(mask)>0:
            map_out[px]= np.mean(mags[mask][(snr[mask]>snrthreshold-1) & (snr[mask]<snrthreshold+1)])
            map_var_out[px]= np.std(mags[mask][(snr[mask]>snrthreshold-1) & (snr[mask]<snrthreshold+1)])
        else:
            map_out[px]=0
            map_var_out[px]= 0
            
    map_out[mask_nans<1]=0
    map_var_out[mask_nans<1]=0

    return map_out, map_var_out

def dr1_method(ra, dec, band, mags, snr, flatSkyGrid, SNRthreshold= 5):
    # follow the paper: choose gals with 4<SNR<6 for 5sigma depth.
    # SNRthreshold= 5 => 5sigma depth.

    depth, depth_std= depth_map_meanSNRrange(ra, dec,  mags= mags,
                                             snr= snr,
                                             snrthreshold= SNRthreshold,
                                             flatSkyGrid= flatSkyGrid)
        
    return depth, depth_std

def get_depth(method,ra,dec,band,arr1,arr2,flatSkyGrid,SNRthreshold=5) :
    """
    Creates a depth map based on the positions and fluxes of a set of objects.
    :param method: method used to compute the depth map. Allowed values: 'dr1', 'desc' and 'fluxerr'.
    :param ra: right ascension for each object.
    :param dec: declination for each object.
    :param band: band for which to compute the depth map.
    :param arr1: measurement of the flux (if using 'fluxerr') or magnitude (otherwise) for each object.
    :param arr2: measurement of the S/N for each object (or `None` if using 'fluxerr').
    :param flatSkyGrid: flatmaps.FlatMapInfo object describing the geometry of the output map.
    :param SNRthreshold: S/N cut to use.
    """
    
    SNRthreshold= int(SNRthreshold)
    print('Creating %s-band %ssigma depth maps'%(band, SNRthreshold))
    if method=='dr1' :
        depth,depth_std=dr1_method(ra,dec,band,mags=arr1,snr=arr2,flatSkyGrid=flatSkyGrid,
                                   SNRthreshold=SNRthreshold)
    elif method=='desc' :
        depth,depth_std=desc_method(ra,dec,band,mags=arr1,snr=arr2,flatSkyGrid=flatSkyGrid,
                                    SNRthreshold=SNRthreshold)
    elif method=='fluxerr' :
        depth,depth_std=flux_err_method(ra,dec,band,flux_err=arr1,flatSkyGrid=flatSkyGrid,
                                        SNRthreshold=SNRthreshold)
    else :
        raise KeyError("Unknown method "+method)

    return depth,depth_std
