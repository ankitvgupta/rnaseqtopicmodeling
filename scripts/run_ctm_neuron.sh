#!/bin/bash

#SBATCH -n 64                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=256000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 24:00:00              #Indicate duration using HH:MM:SS
#SBATCH -p general               #Partition to submit to

#SBATCH -o ./outputs/neuron/ctm_neuron_output_%a.out            #File to which standard out will be written
#SBATCH -e ./outputs/neuron/ctm_neuron_error_%a.err             #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent
cd /scratch
source /n/regal/scrb152/setup.sh
source activate /n/regal/scrb152/Software/Python/pyscrb152
python ~/281FinalProj/RunCTM_neuron.py "${SLURM_ARRAY_TASK_ID}"
