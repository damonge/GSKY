import numpy as np
import os
import sys
import predirs as prd

def submit_job(fname_sql,output_format,do_preview=False,do_download=True,output_file='none') :
    if os.path.isfile(output_file) :
        print("Found "+output_file)
        return

    command="python hscReleaseQuery.py --user=damonge@local"
    command+=" -r pdr1 "
    if do_preview :
        command+=" -p"
    command+=" -f "+output_format
    if do_download :
        command+=" -d"
    command+=" "+fname_sql
    if do_download :
        command+=" > "+output_file
    
    os.system(command)

def add_photoz(mt) :
    sto=""
    sto+="       ,p"+mt+".photoz_mean as pz_mean_"+mt+"\n"
    sto+="       ,p"+mt+".photoz_mode as pz_mode_"+mt+"\n"
    sto+="       ,p"+mt+".photoz_best as pz_best_"+mt+"\n"
    sto+="       ,p"+mt+".photoz_mc as pz_mc_"+mt+"\n"
    return sto

def write_frames(tablename,fname_out,output_format='fits',submit=False,do_download=True) :
    stout="-- Run metadata\n"
    stout+="SELECT *\n"
    stout+="FROM "+tablename+".frame\n"
    stout+=";\n"

    fname_job=None
    if do_download :
        fname_job=prd.predir_saving+tablename.upper()+"_frames."+output_format

    f=open(fname_out,"w")
    f.write(stout)
    f.close()

    if submit :
        submit_job(fname_out,output_format,do_download=do_download,output_file=fname_job)

def write_fieldsearch(tablename,fieldname,fname_out,output_format="fits",
                      submit=False,ra_range=None,exhaustive=False,
                      strict_cuts=False,do_download=True,part=None) :
    filters=['g','r','i','z','y']
    stout="-- Run field, "+fname_out+"\n"

    fname_job='none'
    if do_download :
        fname_job=prd.predir_saving+tablename.upper()+"_"+fieldname.replace('_','').upper()
        if part is not None :
            fname_job+="_part%d"%part
        if not strict_cuts :
            fname_job+="_shearcat"
        fname_job+="_forced."+output_format

    def add_filters(name,behind=True) :
        sthere=""
        for fi in filters :
            if behind :
                sthere+="       ,a."+name+fi+"\n"
            else :
                sthere+="       ,a."+fi+name+"\n"
        return sthere

    stout+="SELECT object_id\n"
    stout+="       ,a.ra as ra\n"
    stout+="       ,a.dec as dec\n"
    stout+="       ,a.tract as tract\n"
    stout+="       ,a.patch as patch\n"
    stout+=add_filters("merge_peak_")
    stout+=add_filters("countinputs",behind=False)
    stout+="       ,a.iflags_pixel_bright_object_center\n"
    stout+="       ,a.iflags_pixel_bright_object_any\n" 
    stout+="       ,a.iclassification_extendedness\n"
    stout+="       ,b.iblendedness_abs_flux as iblendedness_abs_flux\n"
    #Dust extinction
    stout+=add_filters("a_")
    if exhaustive:
        #Psf fluxes and magnitudes
        stout+=add_filters("flux_psf",behind=False)
        stout+=add_filters("flux_psf_err",behind=False)
        stout+=add_filters("flux_psf_flags",behind=False)
        stout+=add_filters("mag_psf",behind=False)
        stout+=add_filters("mag_psf_err",behind=False)
        #Aperture fluxes and magnitudes
        stout+=add_filters("flux_aperture10",behind=False)
        stout+=add_filters("flux_aperture10_err",behind=False)
        stout+=add_filters("flux_aperture_flags",behind=False)
        stout+=add_filters("mag_aperture10",behind=False)
        stout+=add_filters("mag_aperture10_err",behind=False)
    #Cmodel fluxes and magnitudes
    stout+=add_filters("cmodel_flux",behind=False)
    stout+=add_filters("cmodel_flux_err",behind=False)
    stout+=add_filters("cmodel_flux_flags",behind=False)
    stout+=add_filters("cmodel_mag",behind=False)
    stout+=add_filters("cmodel_mag_err",behind=False)
    stout+=add_photoz("eab")
    stout+="       ,c.ishape_hsm_regauss_e1 as ishape_hsm_regauss_e1\n"
    stout+="       ,c.ishape_hsm_regauss_e2 as ishape_hsm_regauss_e2\n"
    stout+="       ,c.ishape_hsm_regauss_sigma as ishape_hsm_regauss_sigma\n"
    stout+="       ,c.ishape_hsm_regauss_resolution as ishape_hsm_regauss_resolution\n"
    stout+="       ,c.ishape_hsm_regauss_flags\n"
    stout+="       ,d.ishape_hsm_regauss_derived_shape_weight as ishape_hsm_regauss_derived_shape_weight\n"
    stout+="       ,d.ishape_hsm_regauss_derived_shear_bias_m as ishape_hsm_regauss_derived_shear_bias_m\n"
    stout+="       ,d.ishape_hsm_regauss_derived_shear_bias_c1 as ishape_hsm_regauss_derived_shear_bias_c1\n"
    stout+="       ,d.ishape_hsm_regauss_derived_shear_bias_c2 as ishape_hsm_regauss_derived_shear_bias_c2\n"
    stout+="       ,d.ishape_hsm_regauss_derived_sigma_e as ishape_hsm_regauss_derived_sigma_e\n"
    stout+="       ,d.ishape_hsm_regauss_derived_rms_e as ishape_hsm_regauss_derived_rms_e\n"    
    stout+="FROM\n"
    stout+="       "+tablename+".forced as a\n"
    stout+="       LEFT JOIN "+tablename+".meas b USING (object_id)\n"
    stout+="       LEFT JOIN "+tablename+".meas2 c USING (object_id)\n"
    stout+="       LEFT JOIN "+tablename+".photoz_ephor_ab peab USING (object_id)\n"
    stout+="       LEFT JOIN "+tablename+".weaklensing_hsm_regauss d USING (object_id)\n"
    stout+="WHERE\n"
    stout+="       a.detect_is_primary=True and\n"
    stout+="       a.icmodel_flags_badcentroid=False and\n"
    stout+="       a.icentroid_sdss_flags=False and\n"
    stout+="       a.iflags_pixel_edge=False and\n"
    stout+="       a.iflags_pixel_interpolated_center=False and\n"
    stout+="       a.iflags_pixel_saturated_center=False and\n"
    stout+="       a.iflags_pixel_cr_center=False and\n"
    stout+="       a.iflags_pixel_bad=False and\n"
    stout+="       a.iflags_pixel_suspect_center=False and\n"
    stout+="       a.iflags_pixel_clipped_any=False and\n"
    stout+="       b.ideblend_skipped=False"
    if strict_cuts :
        stout+=" and\n"
        stout+="       a.gcentroid_sdss_flags=False and\n"
        stout+="       a.rcentroid_sdss_flags=False and\n"
        stout+="       a.zcentroid_sdss_flags=False and\n"
        stout+="       a.ycentroid_sdss_flags=False and\n"
        stout+="       a.gcmodel_flux_flags=False and\n"
        stout+="       a.rcmodel_flux_flags=False and\n"
        stout+="       a.icmodel_flux_flags=False and\n"
        stout+="       a.zcmodel_flux_flags=False and\n"
        stout+="       a.ycmodel_flux_flags=False and\n"
        stout+="       a.gflux_psf_flags=False and\n"
        stout+="       a.rflux_psf_flags=False and\n"
        stout+="       a.iflux_psf_flags=False and\n"
        stout+="       a.zflux_psf_flags=False and\n"
        stout+="       a.yflux_psf_flags=False and\n"
        stout+="       a.gflags_pixel_edge=False and\n"
        stout+="       a.rflags_pixel_edge=False and\n"
        stout+="       a.zflags_pixel_edge=False and\n"
        stout+="       a.yflags_pixel_edge=False and\n"
        stout+="       a.gflags_pixel_interpolated_center=False and\n"
        stout+="       a.rflags_pixel_interpolated_center=False and\n"
        stout+="       a.zflags_pixel_interpolated_center=False and\n"
        stout+="       a.yflags_pixel_interpolated_center=False and\n"
        stout+="       a.gflags_pixel_saturated_center=False and\n"
        stout+="       a.rflags_pixel_saturated_center=False and\n"
        stout+="       a.zflags_pixel_saturated_center=False and\n"
        stout+="       a.yflags_pixel_saturated_center=False and\n"
        stout+="       a.gflags_pixel_cr_center=False and\n"
        stout+="       a.rflags_pixel_cr_center=False and\n"
        stout+="       a.zflags_pixel_cr_center=False and\n"
        stout+="       a.yflags_pixel_cr_center=False and\n"
        stout+="       a.gflags_pixel_bad=False and\n"
        stout+="       a.rflags_pixel_bad=False and\n"
        stout+="       a.zflags_pixel_bad=False and\n"
        stout+="       a.yflags_pixel_bad=False\n"
    stout+=" and\n"
    stout+="       "+tablename+".search_"+fieldname+"(a.skymap_id)"
    if ra_range is not None :
        stout+=" and\n"
        stout+="       a.ra>=%.3lf and\n"%(ra_range[0])
        stout+="       a.ra<%.3lf"%(ra_range[1])
    stout+="\n;\n"

    f=open(fname_out,"w")
    f.write(stout)
    f.close()

    if submit :
        submit_job(fname_out,output_format,do_download=do_download,output_file=fname_job)
