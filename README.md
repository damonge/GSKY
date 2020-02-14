# HSC LSS analyses

This repository contains all the code used to obtain and process the HSC DR1 data to produce tomographic measurements of the galaxy clustering power spectrum.


## Download the raw data

The folder `data_query` contains all the code needed to download the PDR1 data from the HSC [database](https://hsc-release.mtk.nao.ac.jp). Follow the instructions in the [README](./data_query/README.md) there to do so. The main raw products needed are:
* Catalog-level data and per-frame observing condition data.
* Photo-z pdfs for all sources.
* COSMOS-30band photometry data to calibrate the redshift distributions.
* The Arcturus mask.

The data volume is quite large (dominated by the photo-z pdfs), and downloading all of it may take some time.


## The pipeline

The analysis pipeline that processes the raw data to produce power spectrum measurements is provided as a python module in the directory `hsc_lss`. This module is made up of a series of pipeline stages that inherit from the `PipelineStage` class of the `ceci` software. To run the pipeline you therefore first need to install [ceci](https://github.com/LSSTDESC/ceci).

The analysis pipeline runs on each of the 6 HSC WIDE fields (GAMA09H, GAMA15H, HECTOMAP, VVDS, WIDE12H, XMMLSS) individually, and therefore it needs to be run on each of them for each set of configuration parameters. This is done automatically by the script `run_all.py`, which also runs the different versions of the pipeline used in our analysis.

The analysis pipeline consists of 6 stages:
* ReduceCat: takes in the raw catalog data and produces a cleaned version imposing quality cuts, an overall i-magnitude cut and a star-galaxy separation cut. It also produces maps of quantities stored in the forced-photometry catalog: depth, dust absorption in all bands, star density and bright-object mask.
* SystMapper: takes in the per-frame metadata and produces maps of different observing conditions in a given HSC field. The observing conditions mapped are: CCD temperature, airmass, exposure time, sky level, sky sigma, seeing, ellipticity and # of visits.
* PDFMatch: associates each object in the reduced catalog produced by ReduceCat with its photo-z pdf for 5 different photo-z codes (demp, ephor, ephor_ab, frankenz and nnpz).
* COSMOSWeight: processes the COSMOS-30band data and produces colour-space weights for each of those objects so they can be used to produce predictions for the redshift distributions.
* CatMapper: takes in the clean catalog data and bins it into photo-z bins, producing maps of the galaxy density and the corresponding N(z) for each redshift bin (using both COSMOS-30band and pdf stacks from all photo-z codes).
* PowerSpecter: takes the number density maps, mask data and systematics maps to produce measurements of the projected galaxy power spectrum and its covariance matrix with and without deprojection over observational systematics.

The param and configuration files for the different HSC fields are stored in `hsc_lss_params`. All fields use the same common set of configuration parameters, but different paths must be provided to their corresponding raw data files and output directories. See [in_aegis.yml](./hsc_lss_params/in_aegis.yml) and [config.yml](./hsc_lss_params/config.yml) to see the different parameters and options.


## Legacy code

The different scripts, notebooks and previous versions of the pipeline that have contributed towards the final pipeline are stored in the directory `legacy_code`.
