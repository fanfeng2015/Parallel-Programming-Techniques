#!/bin/bash
#SBATCH --partition=cpsc424
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000mb
#SBATCH --job-name=Serial
#SBATCH --time=15:00

module load Langs/Intel/15.0.2 MPI/OpenMPI/2.1.1-intel15
pwd
echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS_PER_NODE
make clean
make serial
time ./serial
