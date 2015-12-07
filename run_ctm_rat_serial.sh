#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=20000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 24:00:00              #Indicate duration using HH:MM:SS
#SBATCH -p serial_requeue               #Partition to submit to

#SBATCH -o ./runctm_output_rat_serial.out            #File to which standard out will be written
#SBATCH -e ./runctm_error_rat_serial.err             #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

#This job will print the name of the host on which the job is executed, and exit
cd /scratch
source /n/regal/scrb152/setup.sh
source activate /n/regal/scrb152/Software/Python/pyscrb152
python ~/281FinalProj/RunCTM_ratorgans_serial.py
