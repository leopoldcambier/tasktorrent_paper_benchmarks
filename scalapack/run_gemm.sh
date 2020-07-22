#!/bin/bash
#SBATCH -o scalapack_gemm_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
mpirun -n ${SLURM_NTASKS} ./pdgemm ${MATRIX_SIZE} 256 ${NROWS} ${NCOLS}
