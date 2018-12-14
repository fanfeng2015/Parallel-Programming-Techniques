#!/bin/bash
#SBATCH --partition=cpsc424
# set total number of MPI processes
#SBATCH --ntasks=8
# set number of MPI processes per node
# (number of nodes is calculated by Slurm)
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-node=2
# set number of cpus per MPI process
#SBATCH --cpus-per-task=1
# set memory per cpu
#SBATCH --mem-per-cpu=4000mb
#SBATCH --job-name=MPI_RUN
#SBATCH --time=15:00

module load Langs/Intel/15 MPI/OpenMPI/2.1.1-intel15
pwd
echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS_PER_NODE
make clean
make mpi-load-balance
time mpirun -n 8 ./mpi-load-balance 8000

# The mpirun command will pick up required info on nodes and cpus from Slurm. 

# You can use mpirun's -n option to reduce the number of MPI processes started 
# on the cpus. (At most 1 MPI proc per Slurm task.)

# You can use mpirun options to control the layout of MPI processes---e.g., to 
# spread processes out onto multiple nodes.

#!/bin/bash
#SBATCH --partition=cpsc424
# set total number of MPI processes
#SBATCH --ntasks=4
# set number of MPI processes per node
# (number of nodes is calculated by Slurm)
#SBATCH --ntasks-per-node=2
# set number of cpus per MPI process
#SBATCH --cpus-per-task=1
# set memory per cpu
#SBATCH --mem-per-cpu=4000mb
#SBATCH --job-name=MPI_RUN
#SBATCH --time=15:00

# time mpirun -n 2 ./task2

# In this example, we've asked Slurm for 4 tasks (2 each on 2 nodes), but we've
# asked mpirun for two MPI procs, which will go onto 1 node. (If "-n 2" is omitted,
# you'll get 4 MPI processes -- 1 per Slurm task.)
