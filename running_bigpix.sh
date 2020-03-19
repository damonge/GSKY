#!/bin/bash

for f in AEGIS GAMA09H GAMA15H HECTOMAP WIDE12H VVDS XMM
do
    fh=$f
    fl="${f,,}"
    
    python -m gsky ReduceCat   --raw_data=./gsky_params/input_list_${fl}.txt   --clean_catalog=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//clean_catalog.fits   --dust_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//dust_map.fits   --star_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//star_map.fits   --bo_mask=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//bo_mask.fits   --masked_fraction=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//masked_fraction.fits   --depth_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//depth_map.fits   --ePSF_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//ePSF_map.fits   --ePSFres_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//ePSFres_map.fits   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    #python -m gsky SystMapper   --frames_data=/global/cscratch1/sd/damonge/GSKY/HSC_data/S16A_WIDE_frames.fits   --masked_fraction=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//masked_fraction.fits   --ccdtemp_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//ccdtemp_maps.fits   --airmass_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//airmass_maps.fits   --exptime_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//exptime_maps.fits   --skylevel_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//skylevel_maps.fits   --sigma_sky_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//sigma_sky_maps.fits   --seeing_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//seeing_maps.fits   --ellipt_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//ellipt_maps.fits   --nvisit_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//nvisit_maps.fits   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    #python -m gsky PDFMatch   --clean_catalog=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//clean_catalog.fits   --pdf_dir=/global/cscratch1/sd/damonge/GSKY/HSC_data/pzs/${fh}/   --pdf_matched=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//pdf_matched.txt   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    python -m gsky COSMOSWeight   --cosmos_data=/global/cscratch1/sd/damonge/GSKY/HSC_data/COSMOS2015_Laigle+_v1.1.fits   --cosmos_hsc=/global/cscratch1/sd/damonge/GSKY/HSC_data/HSC_DEEP_COSMOS.fits   --cosmos_weights=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//cosmos_weights.fits   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    python -m gsky GalMapper   --clean_catalog=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//clean_catalog.fits   --masked_fraction=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//masked_fraction.fits   --cosmos_weights=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//cosmos_weights.fits   --pdf_matched=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//pdf_matched.txt   --ngal_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//ngal_maps.fits   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    python -m gsky ShearMapper   --clean_catalog=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//clean_catalog.fits   --masked_fraction=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//masked_fraction.fits   --cosmos_weights=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//cosmos_weights.fits   --pdf_matched=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//pdf_matched.txt   --gamma_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//gamma_maps.fits   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    python -m gsky ACTMapper   --masked_fraction=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//masked_fraction.fits   --act_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci//act_maps.fits   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml

    #python -m gsky PowerSpecter   --masked_fraction=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/masked_fraction.fits   --shear_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/gamma_maps.fits   --dust_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/dust_map.fits   --star_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/star_map.fits   --depth_map=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/depth_map.fits   --ccdtemp_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/ccdtemp_maps.fits   --airmass_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/airmass_maps.fits   --exptime_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/exptime_maps.fits   --skylevel_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/skylevel_maps.fits   --sigma_sky_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/sigma_sky_maps.fits   --seeing_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/seeing_maps.fits   --ellipt_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/ellipt_maps.fits   --nvisit_maps=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/nvisit_maps.fits   --cosmos_weights=/global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/cosmos_weights.fits   --syst_masking_file=./gsky_params/systematic_cuts/WIDE_WIDE12H_syst_cuts.txt   --dummy=/global/cscratch1/sd/anicola/DATA/HSCxACT/HSC/HSC_processed/${fh}_ceci/dummy   --config=./gsky_params/yaml_files_bigpix/${fl}_config.yml --ngal_maps /global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/ngal_maps.fits --act_maps /global/cscratch1/sd/damonge/GSKY/outputs_bigpix/${fh}_ceci/act_maps.fits
done
