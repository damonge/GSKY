# Downloading the data

The scripts in this directory can be used to download the raw data from the PDR1 database and everything else needed for this pipeline. 
This is done by running `python get_data.py`. A few points must be borne in mind first:
1. Edit the paths in `predirs.py` to point to where you want to save the data (`predir_saving`) and where the Arcturus mask (`arcturus_predir`) is stored.
2. You need to download and install the Arcturus mask and associated code ([Coupon et al. 2017](https://arxiv.org/abs/1705.00622)). This can be found [here](ftp://obsftp.unige.ch/pub/coupon/brightStarMasks/HSC-SSP/HSC-SSP_brightStarMask_Arcturus.tgz).

This script does four things:
1. Download all the catalog-level data from the PDR1 database needed for this pipeline.
2. Add additional information to the downloaded files about the Arcturus mask.
3. Download the COSMOS 30-band photometry data ([Laigle et al. 2016](https://arxiv.org/abs/1604.02350)).
4. Download the photo-z pdfs for all sources in the fields analyzed.

Running everything takes quite some time (O(1-2 days)).
