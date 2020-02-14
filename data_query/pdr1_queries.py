import numpy as np
import os
import sys
import predirs as prd

def submit_job(fname_sql,output_format,do_preview=False,do_download=True,output_file='none') :
    if os.path.isfile(output_file) :
        print("Found "+output_file)
        return

    command="python hscReleaseQuery.py --user=damonge@local"
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

def write_fieldsearch(tablename,fieldname,fname_out,output_format="fits",do_field=True,submit=False,ra_range=None,
                      do_photoz=False,strict_cuts=False,do_download=True,part=None) :
    filters=['g','r','i','z','y']
    stout="-- Run field, "+fname_out+"\n"

    fname_job=None
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
                sthere+="       ,forced."+name+fi+"\n"
            else :
                sthere+="       ,forced."+fi+name+"\n"
        return sthere

    stout+="SELECT object_id\n"
    stout+="       ,forced.ra as ra\n"
    stout+="       ,forced.dec as dec\n"
    stout+="       ,forced.tract as tract\n"
    stout+="       ,forced.patch as patch\n"
    stout+=add_filters("merge_peak_")
    stout+=add_filters("countinputs",behind=False)
    stout+="       ,forced.iflags_pixel_bright_object_center\n"
    stout+="       ,forced.iflags_pixel_bright_object_any\n" 
    stout+="       ,forced.iclassification_extendedness\n"
    stout+="       ,meas.iblendedness_abs_flux as iblendedness_abs_flux\n"
    #Dust extinction
    stout+=add_filters("a_")
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
    if do_photoz :
        stout+=add_photoz("eab")
        stout+=add_photoz("frz")
        stout+=add_photoz("nnz")
    stout+="FROM\n"
    stout+="       "+tablename+".forced as forced\n"
    stout+="       LEFT JOIN "+tablename+".meas meas USING (object_id)\n"
    if do_photoz :
        stout+="       LEFT JOIN "+tablename+".photoz_ephor_ab peab USING (object_id)\n"
        stout+="       LEFT JOIN "+tablename+".photoz_frankenz pfrz USING (object_id)\n"
        stout+="       LEFT JOIN "+tablename+".photoz_nnpz pnnz USING (object_id)\n"
    stout+="WHERE\n"
    stout+="       forced.detect_is_primary=True and\n"
    stout+="       forced.icmodel_flags_badcentroid=False and\n"
    stout+="       forced.icentroid_sdss_flags=False and\n"
    stout+="       forced.iflags_pixel_edge=False and\n"
    stout+="       forced.iflags_pixel_interpolated_center=False and\n"
    stout+="       forced.iflags_pixel_saturated_center=False and\n"
    stout+="       forced.iflags_pixel_cr_center=False and\n"
    stout+="       forced.iflags_pixel_bad=False and\n"
    stout+="       forced.iflags_pixel_suspect_center=False and\n"
    stout+="       forced.iflags_pixel_clipped_any=False and\n"
    stout+="       meas.ideblend_skipped=False"
    if strict_cuts :
        stout+=" and\n"
        stout+="       forced.gcentroid_sdss_flags=False and\n"
        stout+="       forced.rcentroid_sdss_flags=False and\n"
        stout+="       forced.zcentroid_sdss_flags=False and\n"
        stout+="       forced.ycentroid_sdss_flags=False and\n"
        stout+="       forced.gcmodel_flux_flags=False and\n"
        stout+="       forced.rcmodel_flux_flags=False and\n"
        stout+="       forced.icmodel_flux_flags=False and\n"
        stout+="       forced.zcmodel_flux_flags=False and\n"
        stout+="       forced.ycmodel_flux_flags=False and\n"
        stout+="       forced.gflux_psf_flags=False and\n"
        stout+="       forced.rflux_psf_flags=False and\n"
        stout+="       forced.iflux_psf_flags=False and\n"
        stout+="       forced.zflux_psf_flags=False and\n"
        stout+="       forced.yflux_psf_flags=False and\n"
        stout+="       forced.gflags_pixel_edge=False and\n"
        stout+="       forced.rflags_pixel_edge=False and\n"
        stout+="       forced.zflags_pixel_edge=False and\n"
        stout+="       forced.yflags_pixel_edge=False and\n"
        stout+="       forced.gflags_pixel_interpolated_center=False and\n"
        stout+="       forced.rflags_pixel_interpolated_center=False and\n"
        stout+="       forced.zflags_pixel_interpolated_center=False and\n"
        stout+="       forced.yflags_pixel_interpolated_center=False and\n"
        stout+="       forced.gflags_pixel_saturated_center=False and\n"
        stout+="       forced.rflags_pixel_saturated_center=False and\n"
        stout+="       forced.zflags_pixel_saturated_center=False and\n"
        stout+="       forced.yflags_pixel_saturated_center=False and\n"
        stout+="       forced.gflags_pixel_cr_center=False and\n"
        stout+="       forced.rflags_pixel_cr_center=False and\n"
        stout+="       forced.zflags_pixel_cr_center=False and\n"
        stout+="       forced.yflags_pixel_cr_center=False and\n"
        stout+="       forced.gflags_pixel_bad=False and\n"
        stout+="       forced.rflags_pixel_bad=False and\n"
        stout+="       forced.zflags_pixel_bad=False and\n"
        stout+="       forced.yflags_pixel_bad=False\n"
    if do_field :
        stout+=" and\n"
        stout+="       "+tablename+".search_"+fieldname+"(object_id)"
    if ra_range is not None :
        stout+=" and\n"
        stout+="       forced.ra>=%.3lf and\n"%(ra_range[0])
        stout+="       forced.ra<%.3lf"%(ra_range[1])
    stout+="\n;\n"

    f=open(fname_out,"w")
    f.write(stout)
    f.close()

    if submit :
        submit_job(fname_out,output_format,do_download=do_download,output_file=fname_job)
