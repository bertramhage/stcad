#!/bin/sh

#BSUB -J tune_cure

#BSUB -q hpc

#BSUB -n 4

#BSUB -R "rusage[mem=8GB]"

#BSUB -W 6:00

#BSUB -N
#BSUB -B

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.clustering.tests.tune_cure --data_path /dtu/blackhole/10/178320/preprocessed_2024/final/train --sample_size 10000