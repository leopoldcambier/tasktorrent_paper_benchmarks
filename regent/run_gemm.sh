#!/bin/bash
#SBATCH -o legion_dgemm_%j.out
#SBATCH -t 00:15:00

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname

let MATRIX_SIZE=BLOCK_SIZE*NUM_BLOCKS
echo ">>>>slurm_id=${SLURM_JOB_ID},matrix_size=${MATRIX_SIZE},num_blocks=${NUM_BLOCKS},block_size=${BLOCK_SIZE},num_ranks=${SLURM_NTASKS},num_cpus=${NUM_CPUS}"

LAUNCHER="mpirun -n ${SLURM_NTASKS}" ~/legion/language/regent.py my_dgemm.rg -fflow 0 -level 5 -n ${MATRIX_SIZE} -p ${NUM_BLOCKS} -ll:csize 26384 -foverride-demand-index-launch 1 -ll:cpu ${NUM_CPUS} -ll:util ${NUM_CPUS}
