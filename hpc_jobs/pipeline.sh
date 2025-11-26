#!/bin/sh

#BSUB -J prec_pipeline

#BSUB -q hpc

#BSUB -n 32

#BSUB -R "rusage[mem=8GB]"

#BSUB -W 6:00

#BSUB -N
#BSUB -B

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

#python -m src.preprocessing.csv2pkl \
#    --input_dir /dtu/blackhole/10/178320/ais_2024/ \
#    --output_dir /dtu/blackhole/10/178320/preprocessed_2024_2/pickle/ \
#    --run_name pipeline_2024_csv2pkl

python -m src.preprocessing.map_reduce \
    --input_dir /dtu/blackhole/10/178320/preprocessed_2024_2/pickle/ \
    --output_dir /dtu/blackhole/10/178320/preprocessed_2024_2/final/ \
    --num_workers 64 \
    --run_name pipeline_2024_map_reduce

python -m src.preprocessing.train_test_split \
    --data_dir /dtu/blackhole/10/178320/preprocessed_2024_2/final/ \
    --val_size 0.1 \
    --test_size 0.1 \
    --copy