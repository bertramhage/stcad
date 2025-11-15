#!/bin/sh

#BSUB -J download_ais_2024

#BSUB -q hpc

#BSUB -n 4

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 24:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

module swap python3/3.11.9

python src/preprocessing/download.py --from_date 2024-03-01 \
	--to_date 2024-12-31 \
	--destination_path /dtu/blackhole/10/178320/ais_2024
