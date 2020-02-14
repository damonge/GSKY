import os
import numpy as np

def mk_config(cov_type='ana',noi_type='ana',w_ssc=True,msk_syst=False,
              dpj_level=1,dpj_bands=True,mask_type='sirius',cl_theory=False) :
    suffix='cov'+cov_type
    suffix+='_noi'+noi_type+'_msk'+mask_type
    if cl_theory:
        suffix+='_cltheory'
    if w_ssc :
        suffix+='_wssc'
    else :
        suffix+='_wossc'
    if msk_syst :
        suffix+='_msksyst1'
    else :
        suffix+='_msksyst0'
    suffix+='_dpj%d'%dpj_level
    if dpj_bands :
        suffix+='_dpb_bands1'
    else :
        suffix+='_dpb_bands0'
    
    config_name='hsc_lss_params/config_'+suffix+'.yml'

    stout=""
    stout+="global:\n"
    stout+="    depth_cut: 24.5\n"
    stout+="    res: 0.01\n"
    stout+="    pad: 0.1\n"
    stout+="    res_bo: 0.003\n"
    stout+="    flat_project: CAR\n"
    stout+="    mask_type: "+mask_type+"\n"
    stout+="    band: i\n"
    stout+="\n"
    stout+="ReduceCat:\n"
    stout+="    min_snr: 10.\n"
    stout+="    depth_method: fluxerr\n"
    stout+="\n"
    stout+="SystMapper:\n"
    stout+="    hist_nbins: 40\n"
    stout+="\n"
    stout+="COSMOSWeight:\n"
    stout+="    n_neighbors: 20\n"
    stout+="\n"
    stout+="CatMapper:\n"
    stout+="    pz_code: ephor_ab\n"
    stout+="    pz_mark: best\n"
    stout+="    pz_bins: [0.15,0.50,0.75,1.00,1.50]\n"
    stout+="    nz_bin_num: 100\n"
    stout+="    nz_bin_max: 4.0\n"
    stout+="\n"
    stout+="PowerSpecter:\n"
    stout+="    ell_bpws: [100.0,200.0,300.0,400.0,600.0,800.0,1000.0,1400.0,1800.0,2200.0,3000.0,3800.0,4600.0,6200.0,7800.0,9400.0,12600.0,15800.0]\n"
    if cov_type=='ana' :
        stout+="    gaus_covar_type: analytic\n"
    elif cov_type=='sim' :
        stout+="    gaus_covar_type: gaus_sim\n"
    else :
        raise ValueError("Unkonwn covar method")
    if noi_type=='ana' :
        stout+="    noise_bias_type: analytic\n"
    elif noi_type=='sim' :
        stout+="    noise_bias_type: pois_sim\n"
    else :
        raise ValueError("Unkonwn noise method")
    if cl_theory:
        stout+="    guess_spectrum: /global/homes/d/damonge/LSST/LSS_HSC/HyperSupremeStructure-HSC-LSS/hsc_lss_params/cls_guess.txt\n"
    else:
        stout+="    guess_spectrum: NONE\n"
    if w_ssc :
        stout+="    add_ssc: True\n"
    else :
        stout+="    add_ssc: False\n"
    stout+="    z_bias_nodes: [0.0,0.5,0.8,1.1,1.4,4.0]\n"
    stout+="    b_bias_nodes: [1.2,1.3,1.2,1.4,1.6,2.6]\n"
    stout+="    mask_thr: 0.5\n"
    if msk_syst :
        stout+="    mask_systematics: True\n"
    else :
        stout+="    mask_systematics: False\n"
    if dpj_level==0 :
        stout+="    oc_dpj_list: [airmass,seeing,sigma_sky]\n"
    elif dpj_level==1 :
        stout+="    oc_dpj_list: [airmass,ccdtemp,ellipt,exptime,nvisit,seeing,sigma_sky,skylevel]\n"
    else :
        raise ValueError("Unkonwn deprojection level")
    if dpj_bands :
        stout+="    oc_all_bands: True\n"
    else :
        stout+="    oc_all_bands: False\n"
    stout+="    ssc_response_prefix: hsc_lss_params/Responses/Response\n"

    f=open(config_name,"w")
    f.write(stout)
    f.close()

    return suffix,config_name

def run_pipe_all(conf,suffix) :
    cmd='cp '+conf+' hsc_lss_params/config.yml'
    #print(cmd)
    os.system(cmd)
    for field in ['gama09h','gama15h','hectomap','vvds','wide12h','xmmlss'] :
        dirname="/global/cscratch1/sd/damonge/HSC_ceci/WIDE_"+field.upper()+'_sirius_out'
        #Create output directory if not present
        cmd="mkdir -p "+dirname+"/logs/"
        #print(cmd)
        os.system(cmd)

        '''
        #Clean up previous run
        cmd='rm -f '+dirname+'/*mcm* '
        cmd+=dirname+'/gaucov_sims.npz '
        cmd+=dirname+'/windows_l.npz '
        cmd+=dirname+'/*sacc'
        #print(cmd)
        os.system(cmd)

        #Run pipeline
        cmd='ceci hsc_lss_params/in_'+field+'.yml &>log_'+field+'.txt --dry-run'
        print(cmd)
        #os.system(cmd)

        '''
        #Move output
        dirend=dirname+'/'+suffix
        cmd='mkdir -p '+dirend
        #print(cmd)
        os.system(cmd)

        cmd='mv '+dirname+'/*mcm* '
        cmd+=dirname+'/gaucov_sims.npz '
        cmd+=dirname+'/windows_l.npz '
        cmd+=dirname+'/*sacc '
        cmd+=dirend
        #print(cmd)
        os.system(cmd)

##Original settings
#suff,conf=mk_config(dpj_level=0,dpj_bands=False,mask_type='sirius')
#run_pipe_all(conf,suff)
##Fiducial settings
#suff,conf=mk_config(mask_type='sirius')
#run_pipe_all(conf,suff)
##Deproject only main contaminants, use theory power spectra
suff,conf=mk_config(dpj_level=0,dpj_bands=True,mask_type='sirius',cl_theory=True)
run_pipe_all(conf,suff)
##Deproject only main contaminants
#suff,conf=mk_config(dpj_level=0,dpj_bands=True,mask_type='sirius')
#run_pipe_all(conf,suff)
##Simulated covariance
#suff,conf=mk_config(cov_type='sim',mask_type='sirius')
#run_pipe_all(conf,suff)
##Simulated covariance, only main contaminants
#suff,conf=mk_config(cov_type='sim',dpj_level=0,dpj_bands=True,mask_type='sirius')
#run_pipe_all(conf,suff)

##Deproject only main contaminants, mask systematics
#suff,conf=mk_config(dpj_level=0,dpj_bands=True,mask_type='sirius',msk_syst=True)
#run_pipe_all(conf,suff)
#Mask systematics
#suff,conf=mk_config(dpj_level=1,dpj_bands=True,mask_type='sirius',msk_syst=True)
#run_pipe_all(conf,suff)

'''
#Do not include the SSC
suff,conf=mk_config(w_ssc=False)
run_pipe_all(conf,suff)
#Mask systematics
suff,conf=mk_config(msk_syst=True)
run_pipe_all(conf,suff)
#Simulated covariance
suff,conf=mk_config(cov_type='sim')
run_pipe_all(conf,suff)
#Simulated noise bias
suff,conf=mk_config(noi_type='sim')
run_pipe_all(conf,suff)
##Arcturus mask
##This one affects more than just PowerSpecter
#suff,conf=mk_config(mask_type='arcturus')
#run_pipe_all(conf,suff)
'''
