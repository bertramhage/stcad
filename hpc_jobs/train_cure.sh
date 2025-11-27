#!/bin/sh

#BSUB -J train_cure_1

#BSUB -q hpc

#BSUB -n 8

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 6:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/train_cure \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.24 --linkage average --sample_size 1000

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/train_cure \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.28 --linkage average --sample_size 1000