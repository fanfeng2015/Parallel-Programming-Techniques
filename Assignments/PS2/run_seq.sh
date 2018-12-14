#!/bin/bash

echo -e "\n==================== make =============================="
make

echo -e "\n============ Task 1: Serial Program ===================="
./mandseq

echo -e "\n================== clean up ============================"
make clean

echo -e "\n"
