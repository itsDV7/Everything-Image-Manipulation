#!/bin/bash

#SBATCH --job-name=llmslurm    # create a short name for your job
#SBATCH --partition=gpuA100x8
#SBATCH --account=bbug-delta-gpu
#SBATCH --time=00:30:00        # total run time limit (HH:MM:SS)
#SBATCH --output=llmout.log    # optionally specify output file. If not will be written to slurm-jobid.out
#SBATCH --error=llmerr.log     # optionally specify stderr file. If not will be written to slurm-jobid.out
#SBATCH --nodes=1              # node count
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --cpus-per-task=32     # cpu-cores per task (>1 if multi-threaded tasks)

#module load anaconda3_gpu
#conda activate venv

echo "python path `which python3`"

echo "job is starting on `hostname`"
#srun python3 llm.py

srun python3 findoff.py -i "data/list/sci.i.list" -o "out/test.i.offset_b8" -v 1 --useTAN --fluxscale "data/list/flx.i.list"
srun python3 fitoff.py -i "out/test.i.offset_b8" -o "out/test.i.zoff_b8" -b -v 2
srun python3 zoff_apply.py -i "out/test.i.zoff_b8" --fluxscale "data/list/flx.i.list" -o "out/"

# (optionaly) keep node alive for full amount of $TIME
#sleep infinity
