#!/bin/bash

echo -e "\n==================== make ================================================"
make 

echo -e "\n==================== Exercise 1 - Compiler Option (a) ===================="
./ex1_a

echo -e "\n==================== Exercise 1 - Compiler Option (b) ===================="
./ex1_b

echo -e "\n==================== Exercise 1 - Compiler Option (c) ===================="
./ex1_c

echo -e "\n==================== Exercise 1 - Compiler Option (d) ===================="
./ex1_d

echo -e "\n==================== Exercise 1 - Division Operation Latency ============="
./ex1_div

echo -e "\n==================== Exercise 2 - Vector Triad Performance ==============="
./ex2

echo -e "\n==================== clean up ============================================"
make clean

echo -e "\n"
