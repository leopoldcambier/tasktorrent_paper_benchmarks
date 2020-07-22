#!/bin/bash
#SBATCH -o scalapack_chol_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
mpirun -n ${SLURM_NTASKS} ./pdpotrf ${MATRIX_SIZE} ${BLOCK_SIZE} ${NPROWS} ${NPCOLS}
