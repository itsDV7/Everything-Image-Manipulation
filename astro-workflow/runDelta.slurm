#!/bin/bash

#SBATCH --job-name=llmslurm
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbug-delta-gpu
#SBATCH --output=llmout.log    # optionally specify output file. If not will be written to slurm-jobid.out
#SBATCH --error=llmerr.log     # optionally specify stderr file. If not will be written to slurm-jobid.out
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=10g

# export local environment variables to the job
#SBATCH --export=ALL

echo "python path `which python3`"

echo "job is starting on `hostname`"
#srun python3 llm.py

srun python3 findoff.py -i "data/list/sci.i.list" -o "out/test.i.offset_b8" -v 1 --useTAN --fluxscale "data/list/flx.i.list"
srun python3 fitoff.py -i "out/test.i.offset_b8" -o "out/test.i.zoff_b8" -b -v 2
srun python3 zoff_apply.py -i "out/test.i.zoff_b8" --fluxscale "data/list/flx.i.list" -o "out/"

# (optionaly) keep node alive for full amount of $TIME
#sleep infinity
