#!/bin/sh

#BSUB -J train_cure

#BSUB -q hpc

#BSUB -n 4

#BSUB -R "rusage[mem=8GB]"

#BSUB -W 2:00

#BSUB -N
#BSUB -B

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/data/models/cure_model \
    --sample_size 1000 \
    --n_representatives 20 \
    --compression 0.2 \
    --linkage single \
    --dendrogram_p 100