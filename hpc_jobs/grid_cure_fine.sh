#!/bin/sh
#BSUB -J tune_cure_fine
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 16:00
#BSUB -N
#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project
. .venv/bin/activate

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.05 --assignment_threshold 0.20

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.05 --assignment_threshold 0.22

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.05 --assignment_threshold 0.24

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.05 --assignment_threshold 0.26

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.05 --assignment_threshold 0.28

python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.05 --assignment_threshold 0.30

# Try no phase 1 pruning
python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --pruning_trigger 0.00 --assignment_threshold 0.26

# No phase 2 pruning
python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --phase2_ratio 0.00 --pruning_trigger 0.05  --assignment_threshold 0.26

# No pruning at all
python -m src.clustering.hierarchical \
    --data_path /zhome/ea/6/187439/computational-tools-project/data/embeddings.npz \
    --output_path /zhome/ea/6/187439/computational-tools-project/locals/test_cure \
    --compression 0.4 --phase2_ratio 0.00 --pruning_trigger 0.00  --assignment_threshold 0.26