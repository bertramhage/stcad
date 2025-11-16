#!/bin/sh

#BSUB -J prec_pipeline

#BSUB -q hpc

#BSUB -n 64

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 72:00

#BSUB -N
#BSUB -B

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.preprocessing.map_reduce \
    --input_dir /dtu/blackhole/10/178320/preprocessed_2024/pickle/ \
    --output_dir /dtu/blackhole/10/178320/preprocessed_2024/final/ \
    --num_workers 128 \
    --run_name pipeline_2024_map_reduce
