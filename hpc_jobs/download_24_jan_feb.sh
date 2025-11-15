#!/bin/sh

#BSUB -J download_ais_2024_batch

#BSUB -q hpc

#BSUB -n 4

#BSUB -R "rusage[mem=32GB]"

#BSUB -W 1:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python locals/download_batch.py --destination_path /dtu/blackhole/10/178320/ais_2024
