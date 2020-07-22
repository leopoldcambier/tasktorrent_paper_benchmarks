#!/bin/bash

NROWS=1 NCOLS=1 MATRIX_SIZE=8192  sbatch -c 32 -n 1 run_gemm.sh
NROWS=1 NCOLS=1 MATRIX_SIZE=16384 sbatch -c 32 -n 1 run_gemm.sh

NROWS=4 NCOLS=2 MATRIX_SIZE=8192  sbatch -c 32 -n 8 run_gemm.sh
NROWS=4 NCOLS=2 MATRIX_SIZE=16384 sbatch -c 32 -n 8 run_gemm.sh
NROWS=4 NCOLS=2 MATRIX_SIZE=32768 sbatch -c 32 -n 8 run_gemm.sh

NROWS=8 NCOLS=8 MATRIX_SIZE=8192  sbatch -c 32 -n 64 run_gemm.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=16384 sbatch -c 32 -n 64 run_gemm.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 sbatch -c 32 -n 64 run_gemm.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=65536 sbatch -c 32 -n 64 run_gemm.sh
