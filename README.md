# rnaseqtopicmodeling
## Ankit Gupta, Harvard University
A repository for research work involving topic modeling RNA-seq data.


## Overview
This repository contains work involving applying various topic modeling algorithms to RNA-seq data. The goal is to investigation whether topic modeling can be used to conduct hypothesis-free analysis of RNA-seq data, and ultimately to conduct biological inference. 

## A quick note
Many of the files in this repository are meant for a cluster computing environment. For the purposes of this work, much of this was run on Harvard University's Odyssey Research Cluster.

## Important Files
- [CTM.py](CTM.py): An implementation of Correlated Topic Models (Serial).
- [CTMParalle.py](CTMParallel.py): A parallelized version of CTM using python multiprocessing. This was used for the majority of the simultations in this research, and was run on 64-core nodes on Harvard's Odyssey cluster.
- Files beginning with RunCTM_*: These files were used to load the various data files, and run the appriopriate CTM algorithm
- Files matching run_ctm_*.sh: These files were using by sbatch to run jobs on Harvard's Odyssey cluster. 

