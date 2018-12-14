#!/bin/bash

echo -e "\n========================= make ================================"
make

echo -e "\n=============== Task 3: OpenMP Program (Tasks) ================"
echo -e "\n=============== 1. tasks by cell =============================="
export OMP_NUM_THREADS=1
./mandomp-tasks
export OMP_NUM_THREADS=2
./mandomp-tasks
export OMP_NUM_THREADS=4
./mandomp-tasks
export OMP_NUM_THREADS=8
./mandomp-tasks

echo -e "\n=============== 2. tasks by row ==============================="
export OMP_NUM_THREADS=1
./mandomp-tasks-row
export OMP_NUM_THREADS=2
./mandomp-tasks-row
export OMP_NUM_THREADS=4
./mandomp-tasks-row
export OMP_NUM_THREADS=8
./mandomp-tasks-row

echo -e "\n=============== 3. tasks by all threads ======================="
export OMP_NUM_THREADS=1
./mandomp-tasks-allthreads
export OMP_NUM_THREADS=2
./mandomp-tasks-allthreads
export OMP_NUM_THREADS=4
./mandomp-tasks-allthreads
export OMP_NUM_THREADS=8
./mandomp-tasks-allthreads

echo -e "\n======================= clean up =============================="
make clean

echo -e "\n"
