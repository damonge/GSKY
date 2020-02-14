from hsc_queries import write_frames, write_fieldsearch
import predirs as prd
import os


################################
#                              #
#  Download catalog-level data #
#                              #
################################

#Per-frame metadata
write_frames("s16a_wide","frames_wide.sql",submit=True)
write_frames("s16a_deep","frames_deep.sql",submit=True)
write_frames("s16a_udeep","frames_udeep.sql",submit=True)

'''
#WIDE fields
for fld in ['aegis','gama09h','gama15h','hectomap','wide12h','xmm_lss'] :
    write_fieldsearch("pdr1_wide",fld,"field_wide_"+fld+"_pz_strict.sql",do_field=True,
                      submit=True,do_photoz=True,strict_cuts=True)
write_fieldsearch("pdr1_wide",'vvds',"field_wide_vvds_h1_pz_strict.sql",do_field=True,
                  submit=True,ra_range=[330.,336.],do_photoz=True,strict_cuts=True,part=1)
write_fieldsearch("pdr1_wide",'vvds',"field_wide_vvds_h2_pz_strict.sql",do_field=True,
                  submit=True,ra_range=[336.,342.],do_photoz=True,strict_cuts=True,part=2)

#DEEP fields
for fld in ['cosmos','deep2_3','elais_n1','xmm_lss'] :
    write_fieldsearch("pdr1_deep",fld,"field_deep_"+fld+"_pz_strict.sql",do_field=True,
                      submit=True,do_photoz=True,strict_cuts=True)

#UDEEP fields
for fld in ['cosmos','sxds'] :
    write_fieldsearch("pdr1_udeep",fld,"field_udeep_"+fld+"_pz_strict.sql",do_field=True,
                      submit=True,do_photoz=True,strict_cuts=True)

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
        
        print 'Downloading COSMOS2015_Laigle+_v1.1.fits.gz...'
        urllib.urlretrieve(url, 'COSMOS2015_Laigle+_v1.1.fits.gz')
        
        print 'Decompressing COSMOS2015_Laigle+_v1.1.fits.gz...'
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

def get_pdfs(fld,pzcode) :
    predir=prd.predir_saving+fld.upper()+'/'+pzcode+'/'
    if os.path.isfile(predir+'done') :
        print("Found pdfs - ("+fld+","+pzcode+")")
        return

    tarfile='pdr1_'+pzcode+'_'+fld+'.tar.xz'
    url='https://hsc-release.mtk.nao.ac.jp/archive/photoz/pdr1/pdf/'+pzcode
    url+='/'+tarfile
    os.system('wget '+url)
    os.system('mkdir -p '+predir)
    os.system('touch '+predir+'done')
    os.system('tar -C '+predir+' -xf '+tarfile)
    os.system('rm '+tarfile)

for pc in ['nnpz','ephor','ephor_ab','demp','frankenz'] :
    for f in ['wide_aegis','wide_gama09h','wide_gama15h','wide_hectomap','wide_vvds',
              'wide_wide12h','wide_xmmlss',
              'deep_cosmos','deep_elaisn1','deep_xmmlss','deep_deep23'] :
        get_pdfs(f,pc)
'''
