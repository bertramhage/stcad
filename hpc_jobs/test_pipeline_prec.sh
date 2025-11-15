#!/bin/sh

#BSUB -J test_prec_pipeline

#BSUB -q hpc

#BSUB -n 8

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 1:00

#BSUB -N

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/computational-tools-project

. .venv/bin/activate

python -m src.preprocessing.download \
    --from_date 2024-05-01 \
	--to_date 2024-05-03 \
	--destination_path /dtu/blackhole/10/178320/prec_test \
    --run_name test_prec_pipeline

python -m src.preprocessing.csv2pkl \
    --input_dir /dtu/blackhole/10/178320/prec_test \
    --output_dir /dtu/blackhole/10/178320/prec_test/pickle/ \
    --run_name test_prec_pipeline

python -m src.preprocessing.map_reduce \
    --input_dir /dtu/blackhole/10/178320/prec_test/pickle/ \
    --output_dir /dtu/blackhole/10/178320/prec_test/processed/ \
    --run_name test_prec_pipeline
