# rnaseqtopicmodeling
## Ankit Gupta, Harvard University
A repository for research work involving topic modeling RNA-seq data.


## Overview
This repository contains work involving applying various topic modeling algorithms to RNA-seq data. The goal is to investigation whether topic modeling can be used to conduct hypothesis-free analysis of RNA-seq data, and ultimately to conduct biological inference. 

## A quick note
Many of the files in this repository are meant for a cluster computing environment. For the purposes of this work, much of this was run on Harvard University's Odyssey Research Cluster.

## Important Files
- [CTMParallel.py](CTMParallel.py): A parallelized version of CTM using python multiprocessing. This was used for the majority of the simulations in this research, and was run on 64-core nodes on Harvard's Odyssey cluster.
- [CTM.py](CTM.py): An implementation of Correlated Topic Models (Serial). This was used prior to developing the parallel version.
- Files matching RunCTM*: These files were used to load the various data files, and run the appriopriate CTM algorithm, for experiments 2 and 3.
- Files matching run_ctm_*.sh: These files were using by sbatch to run jobs on Harvard's Odyssey cluster. 
- [RunTrainedModel.ipynb](RunTrainedModel.ipynb): An ipython notebook where I ran the classification algorithms on the CTM models trained using the above files, for experiments 2 and 3. Note that this file was used primarily on Odyssey.
- [LDA.ipynb](LDA.ipynb): An ipython notebook where I trained the LDA models for experiments 2 and 3.
- [LDA-Classification.ipynb](LDA-Classification.ipynb): An ipython notebook where I ran classification algorithms on the trained LDA models for experiments 2 and 3.
- [ProofofConcept.ipynb](ProofOfConcept.ipynb): An ipython notebook where I did the proof of concept (experiment 1).
