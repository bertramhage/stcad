#!/bin/sh

#BSUB -J csv2pkl_2024

#BSUB -q hpc

#BSUB -n 8

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 48:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.preprocessing.csv2pkl \
    --input_dir /dtu/blackhole/10/178320/ais_2024/ \
    --output_dir /dtu/blackhole/10/178320/preprocessed_2024/pickle/ 