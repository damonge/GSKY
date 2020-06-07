from hsc_queries import write_frames, write_fieldsearch
import predirs as prd
import os


################################
#                              #
#  Download catalog-level data #
#                              #
################################
'''
#Per-frame metadata
write_frames("s16a_wide","frames_wide.sql",submit=True)
write_frames("s16a_deep","frames_deep.sql",submit=True)
write_frames("s16a_udeep","frames_udeep.sql",submit=True)

#WIDE fields
for fld in ['aegis','gama09h','gama15h','hectomap','wide12h','xmm']:
    write_fieldsearch("s16a_wide",fld,"field_wide_"+fld+"_pz_strict.sql",
                      submit=True, strict_cuts=True, do_download=False)
write_fieldsearch("s16a_wide",'vvds',"field_wide_vvds_h1_pz_strict.sql",
                  submit=True,ra_range=[330.,336.],strict_cuts=True,part=1,
                  do_download=False)
write_fieldsearch("s16a_wide",'vvds',"field_wide_vvds_h2_pz_strict.sql",
                  submit=True,ra_range=[336.,342.],strict_cuts=True,part=2,
                  do_download=False)

#DEEP fields
for fld in ['cosmos','deep2_3','elais_n1','xmm_lss'] :
    write_fieldsearch("s16a_deep",fld,"field_deep_"+fld+"_pz_strict.sql",
                      submit=True,strict_cuts=True,do_download=False,
                      w_lensing=False)

#UDEEP fields
for fld in ['cosmos','sxds'] :
    write_fieldsearch("s16a_udeep",fld,"field_udeep_"+fld+"_pz_strict.sql",
                      submit=True,strict_cuts=True,do_download=False,
                      w_lensing=False)

#WIDE-depth COSMOS
for see in ['best','median','worst'] :
    write_fieldsearch("pdr1_cosmos_widedepth_"+see,"none","field_cosmo_wide_"+see+".sql",
                      do_field=False,submit=True,do_photoz=False)

#WIDE fields, shear catalog
for fld in ['aegis','gama09h','gama15h','hectomap','wide12h','xmm_lss'] :
    write_fieldsearch("pdr1_wide",fld,"field_wide_"+fld+"_pz.sql",do_field=True,
                      submit=True,do_photoz=True)
write_fieldsearch("pdr1_wide",'vvds',"field_wide_vvds_h1_pz.sql",do_field=True,submit=True,
                  ra_range=[330.,336.],do_photoz=True,part=1)
write_fieldsearch("pdr1_wide",'vvds',"field_wide_vvds_h2_pz.sql",do_field=True,submit=True,
                  ra_range=[336.,342.],do_photoz=True,part=2)


#############################
#                           #
#  Add Arcturus mask flags  #
#                           #
#############################

def add_Arcturus_flag(fname_in) :
    from astropy.io import fits
    
    names=fits.open(fname_in)[1].data.names
    if 'mask_Arcturus' in names :
        print("Found Arcturus flag "+fname_in)
        return
    else :
        print("NOO "+fname_in)
    
    cmd=prd.venice_exec
    cmd+=" -m "+prd.arcturus_predir+"/reg/masks_all.reg"
    cmd+=" -cat "+fname_in
    cmd+=" -xcol ra -ycol dec -o "+fname_in+".tmp.fits"+" -f all -flagName mask_Arcturus"
    print(cmd)
    os.system(cmd)
    cmd2="mv "+fname_in+".tmp.fits "+fname_in
    print(cmd2)
    os.system(cmd2)

for fld in ['aegis','gama09h','gama15h','hectomap','wide12h','xmm_lss'] :
    fname=prd.predir_saving+'PDR1_WIDE_'+fld.replace('_','').upper()+'_shearcat_forced.fits'
    add_Arcturus_flag(fname)
    fname=prd.predir_saving+'PDR1_WIDE_'+fld.replace('_','').upper()+'_forced.fits'
    add_Arcturus_flag(fname)
for p in [1,2] :
    fname=prd.predir_saving+'PDR1_WIDE_VVDS_part%d_shearcat_forced.fits'%p
    add_Arcturus_flag(fname)
    fname=prd.predir_saving+'PDR1_WIDE_VVDS_part%d_forced.fits'%p
    add_Arcturus_flag(fname)
for fld in ['cosmos','deep2_3','elais_n1','xmm_lss'] :
    fname=prd.predir_saving+'PDR1_DEEP_'+fld.replace('_','').upper()+'_forced.fits'
    add_Arcturus_flag(fname)
for fld in ['cosmos','sxds'] :
    fname=prd.predir_saving+'PDR1_UDEEP_'+fld.replace('_','').upper()+'_forced.fits'
    add_Arcturus_flag(fname)
for see in ['best','median','worst'] :
    fname=prd.predir_saving+'PDR1_COSMOS_WIDEDEPTH_'+see.upper()+'_NONE_shearcat_forced.fits'
    add_Arcturus_flag(fname)

###########################
#                         #
#  Get COSMOS-30band data #
#                         #
###########################

def get_cosmos30band() :
    fname_out=prd.predir_saving+'COSMOS2015_Laigle+_v1.1.fits'
    
    if os.path.isfile(fname_out) :
        print("Found COSMOS data")
        return
    else :
        import urllib
        import gzip
        
        url = 'ftp://ftp.iap.fr/pub/from_users/hjmcc/COSMOS2015/'
        url+= 'COSMOS2015_Laigle+_v1.1.fits.gz'
        
        print('Downloading COSMOS2015_Laigle+_v1.1.fits.gz...')
        urllib.urlretrieve(url, 'COSMOS2015_Laigle+_v1.1.fits.gz')
        
        print('Decompressing COSMOS2015_Laigle+_v1.1.fits.gz...')
        with gzip.open('./COSMOS2015_Laigle+_v1.1.fits.gz', 'rb') as readfile:
            with open('./COSMOS2015_Laigle+_v1.1.fits', 'wb') as writefile:
                gzdata = readfile.read()
                writefile.write(gzdata)

        os.remove('./COSMOS2015_Laigle+_v1.1.fits.gz')
        os.system('mv ./COSMOS2015_Laigle+_v1.1.fits '+fname_out)
get_cosmos30band()


##########################
#                        #
#  Download photo-z pdfs #
#                        #
##########################

def get_pdfs(fld):
    predir=prd.predir_saving+'pzs/'+fld.upper()+'/'
    if os.path.isfile(predir+'done') :
        print("Found pdfs - ("+fld+") "+predir)
        return

    print(predir)
    os.system('mkdir -p '+predir)
    url='https://hsc-release.mtk.nao.ac.jp/archive/filetree/s16a-shape-catalog/Sirius/'+fld.upper()+'_tracts/'
    os.system('wget -np -r --user=damonge --password=$HSC_SSP_CAS_PASSWORD '+url)
    predir_i = 'hsc-release.mtk.nao.ac.jp/archive/filetree/s16a-shape-catalog/Sirius/'+fld.upper()+'_tracts/'
    os.system('rm ' + predir_i + 'index.html')
    os.system('mv '+predir_i+'*.fits '+predir)
    os.system('rm -r '+predir_i)
    os.system('touch '+predir+'done')

#for f in ['aegis','gama09h','gama15h','hectomap','vvds', 'wide12h','xmm']:
for f in ['gama15h']:
    get_pdfs(f)
'''

def get_cosmos_hsc_data(fname, subdir=None):
    predir = prd.predir_saving + 'cosmos_pzs/'
    if subdir is not None:
        predir += subdir + '/'
    os.system('mkdir -p ' + predir)
    url = "https://hsc-release.mtk.nao.ac.jp/archive/filetree/cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/"
    if subdir is not None:
        url += subdir + '/'
    url += fname
    os.system('wget --user=damonge --password=$HSC_SSP_CAS_PASSWORD '+url)
    os.system("mv " + fname + " " + predir)
get_cosmos_hsc_data("Afterburner_reweighted_COSMOS_photoz_FDFC.fits")
get_cosmos_hsc_data("target_wide_s17a_9812.fits")
get_cosmos_hsc_data("target_wide_s17a_9813.fits")
get_cosmos_hsc_data("PDz.target_wide_s17a_9812.cat.fits", subdir="DEmP")
get_cosmos_hsc_data("PDz.target_wide_s17a_9813.cat.fits", subdir="DEmP")
get_cosmos_hsc_data("target_wide_s17a_9812.v1.P.cat.fits", subdir="MLZ")
get_cosmos_hsc_data("target_wide_s17a_9813.v1.P.cat.fits", subdir="MLZ")
get_cosmos_hsc_data("target_wide_s17a_9812_all.cat.fits", subdir="Mizuki")
get_cosmos_hsc_data("target_wide_s17a_9813_all.cat.fits", subdir="Mizuki")
get_cosmos_hsc_data("target_wide_s17a_9812_mags_photoz.cat.fits", subdir="NNPZ")
get_cosmos_hsc_data("target_wide_s17a_9813_mags_photoz.cat.fits", subdir="NNPZ")
get_cosmos_hsc_data("pdf-s17a_wide-9812.cat.fits", subdir="ephor")
get_cosmos_hsc_data("pdf-s17a_wide-9813.cat.fits", subdir="ephor")
get_cosmos_hsc_data("pdf-s17a_wide-9812.cat.fits", subdir="ephor_ab")
get_cosmos_hsc_data("pdf-s17a_wide-9813.cat.fits", subdir="ephor_ab")

'''
for c in ['demp', 'ephor', 'ephor_ab', 'frankenz', 'mizuki', 'mlz', 'nnpz']:
    url = 'https://hsc-release.mtk.nao.ac.jp/archive/filetree/s16a-shape-catalog/pz_pdf_bins_'+c+'.fits'
    os.system('wget --user=damonge --password=$HSC_SSP_CAS_PASSWORD '+url)
    for f in ['aegis','gama09h','gama15h','hectomap','vvds',
              'wide12h','xmm']:
        fname = prd.predir_saving+'pzs/'+f.upper()+'/'+c+'/pz_pdf_bins.fits'
        os.system('cp pz_pdf_bins_'+c+'.fits '+fname)
    os.system('rm pz_pdf_bins_'+c+'.fits')
'''
