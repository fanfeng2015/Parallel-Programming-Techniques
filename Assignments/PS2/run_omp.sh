#!/bin/bash

echo -e "\n========================= make ================================"
make

echo -e "\n========== Task 2: OpenMP Program (Loop Directives) ==========="
echo -e "\n========== 1. Without schedule or collapse ==================="
export OMP_NUM_THREADS=1
./mandomp
export OMP_NUM_THREADS=2
./mandomp
export OMP_NUM_THREADS=4
./mandomp
export OMP_NUM_THREADS=8
./mandomp

echo -e "\n========== 2. schedule(static,1) =============================="
export OMP_SCHEDULE="static,1"
export OMP_NUM_THREADS=1
./mandomp-schedule
export OMP_NUM_THREADS=2
./mandomp-schedule
export OMP_NUM_THREADS=4
./mandomp-schedule
export OMP_NUM_THREADS=8
./mandomp-schedule

echo -e "\n========== 2. schedule(static,10) ============================="
export SCHEDULE="static,10"
export OMP_NUM_THREADS=1
./mandomp-schedule
export OMP_NUM_THREADS=2
./mandomp-schedule
export OMP_NUM_THREADS=4
./mandomp-schedule
export OMP_NUM_THREADS=8
./mandomp-schedule

echo -e "\n========== 2. schedule(dynamic) ==============================="
export SCHEDULE="dynamic"
export OMP_NUM_THREADS=1
./mandomp-schedule
export OMP_NUM_THREADS=2
./mandomp-schedule
export OMP_NUM_THREADS=4
./mandomp-schedule
export OMP_NUM_THREADS=8
./mandomp-schedule

echo -e "\n========== 2. schedule(dynamic,10) ============================"
export SCHEDULE="dynamic,10"
export OMP_NUM_THREADS=1
./mandomp-schedule
export OMP_NUM_THREADS=2
./mandomp-schedule
export OMP_NUM_THREADS=4
./mandomp-schedule
export OMP_NUM_THREADS=8
./mandomp-schedule

echo -e "\n========== 2. schedule(guided) ================================"
export SCHEDULE="guided"
export OMP_NUM_THREADS=1
./mandomp-schedule
export OMP_NUM_THREADS=2
./mandomp-schedule
export OMP_NUM_THREADS=4
./mandomp-schedule
export OMP_NUM_THREADS=8
./mandomp-schedule

echo -e "\n========== 3. schedule(static,1) collapse(2) =================="
export OMP_SCHEDULE="static,1"
export OMP_NUM_THREADS=1
./mandomp-collapse
export OMP_NUM_THREADS=2
./mandomp-collapse
export OMP_NUM_THREADS=4
./mandomp-collapse
export OMP_NUM_THREADS=8
./mandomp-collapse

echo -e "\n========== 3. schedule(static,10) collapse(2) ================="
export SCHEDULE="static,10"
export OMP_NUM_THREADS=1
./mandomp-collapse
export OMP_NUM_THREADS=2
./mandomp-collapse
export OMP_NUM_THREADS=4
./mandomp-collapse
export OMP_NUM_THREADS=8
./mandomp-collapse

echo -e "\n========== 3. schedule(dynamic) collapse(2) ==================="
export SCHEDULE="dynamic"
export OMP_NUM_THREADS=1
./mandomp-collapse
export OMP_NUM_THREADS=2
./mandomp-collapse
export OMP_NUM_THREADS=4
./mandomp-collapse
export OMP_NUM_THREADS=8
./mandomp-collapse

echo -e "\n========== 3. schedule(dynamic,10) collapse(2) ================"
export SCHEDULE="dynamic,10"
export OMP_NUM_THREADS=1
./mandomp-collapse
export OMP_NUM_THREADS=2
./mandomp-collapse
export OMP_NUM_THREADS=4
./mandomp-collapse
export OMP_NUM_THREADS=8
./mandomp-collapse

echo -e "\n========== 3. schedule(guided) collapse(2) ===================="
export SCHEDULE="guided"
export OMP_NUM_THREADS=1
./mandomp-collapse
export OMP_NUM_THREADS=2
./mandomp-collapse
export OMP_NUM_THREADS=4
./mandomp-collapse
export OMP_NUM_THREADS=8
./mandomp-collapse

echo -e "\n======================= clean up =============================="
make clean

echo -e "\n"
