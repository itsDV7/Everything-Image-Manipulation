# Astronomy pipeline

## Overview
The goal of the process is to remove residual background light from the images. The process has 3 steps implemented in 3 python scripts:

1. `findoff.py` is comparing the area of overlap between each image and calculating the median difference.
2. `fitoff.py` is taking the differences among all the overlapping images (from findoff) and calculating the best offset value to add/subtract to the pixel values in each image
3. `zoff_apply.py` is taking the values found in fitoff and then applying them to the original images.
4. `runDelta.slurm` is a script to run the code on the DELTA supercomputer. Parameters required to run it are defined at the top of the script.

The method is based on paper https://ui.adsabs.harvard.edu/abs/1995ASPC...77..335R/abstract.
The code is taken from https://github.com/rohan-uiuc/ncsa-hackathon-workflows/tree/main/astronomy-pipeline.

## Workflow
1. Activate virtual env
2. Install requirements if not already installed
3. Acquire parameters for the python scripts (from the user) and for the slurm script (without relying on the user)
3. For each band (i,r), run `findoff.py`. using the output from `findoff`, run `fitoff` and using the output from `fitoff`, run `zoff_apply` 

### Example usage of the python scripts
1. `python3 findoff.py -i "data/list/sci.i.list" -o "out/test.i.offset_b8" -v 1 --useTAN --fluxscale "data/list/flx.i.list"` .  
2. `python3 fitoff.py -i "out/test.i.offset_b8" -o "out/test.i.zoff_b8" -b -v 2`
3. `python3 zoff_apply.py -i "out/test.i.zoff_b8" --fluxscale "data/list/flx.i.list" -o "out/"`

## Data
- Download and extract data from shared GDrive https://drive.google.com/drive/folders/1mT5QSrH1sv20HYG2P7HDP9L1-KveDJmX
- Place the data folder inside the main directory of this repository.

## Requirements
- Python 3.12+
- pip

## Acknowledgements
- [National Center for Supercomputer Applications](https://www.ncsa.illinois.edu/)
- Robert Gruendl, Sr. Research Scientist, NCSA.
- Dark Energy Survey https://des.ncsa.illinois.edu/, https://github.com/DarkEnergySurvey/despyastro
