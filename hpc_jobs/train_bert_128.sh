#!/bin/sh

### Job Name:
#BSUB -J train_bert

### Queue Name:
#BSUB -q gpua100

### Ensure 80GB GPU used
#BSUB -R "select[gpu80gb]"

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core (min 4 cores pr gpu)
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

#BSUB -W 2:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err


### cd to repo dir
cd ~/computational-tools-project

### activate environment
. .venv/bin/activate

### load python and cuda modules
module swap cuda/12.6.3

### run script
python -m src.sequence_modelling.train_bert \
    --data_dir /dtu/blackhole/10/178320/preprocessed_2024/final \
    --output_dir ./data/models/trajectory_bert_model-128 \
    --run_name bert_2024_128 \
    --num_epochs 30