#!/bin/sh

#BSUB -J test_prec_pipeline

#BSUB -q hpc

#BSUB -n 16

#BSUB -R "rusage[mem=8GB]"

#BSUB -W 48:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.preprocessing.map_reduce \
    --input_dir /dtu/blackhole/10/178320/preprocessed_2024/pickle/ \
    --output_dir /dtu/blackhole/10/178320/preprocessed_2024/final/ \
    --run_name pipeline_2024_map_reduce
