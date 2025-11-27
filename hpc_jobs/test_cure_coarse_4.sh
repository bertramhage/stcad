#!/bin/sh

#BSUB -J tune_cure_coarse_2

#BSUB -q hpc

#BSUB -n 8

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 4:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

## Linkage average
python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.2 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.22 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.24 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.26 --linkage ward

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.28 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.6 --pruning_fraction 0.05 --assignment_threshold 0.30 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.2 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.22 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.24 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.26 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.28 --linkage average

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings_small.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure_5 \
    --compression 0.8 --pruning_fraction 0.05 --assignment_threshold 0.30 --linkage average
