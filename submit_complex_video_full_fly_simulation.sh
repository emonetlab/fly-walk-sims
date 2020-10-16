#!/bin/bash
#SBATCH --job-name=111_10.1
#SBATCH --mem-per-cpu=6000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-99
#SBATCH --output=out.txt
#SBATCH --open-mode=append
#SBATCH --time=5:00:00

python complex_video_full_fly_run_script.py $SLURM_ARRAY_TASK_ID

exit 0
