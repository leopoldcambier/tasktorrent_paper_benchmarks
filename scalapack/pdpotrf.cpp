#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cstdlib>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include "mpi.h"
#include "mkl_cblas.h"

/**
 * Compile with something like
 * mpicxx test_dpotrf.cpp \
 *     -L/.../scalapack/2.1.0/lib \
 *     -lscalapack
 *
 * Or with MKL something like
 *
 * mpiicpc pdpotrf.cpp -o pdpotrf  -I${MKLROOT}/include ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
 *
 * Usage: ./pdpotrf matrix_size block_size nprocs_row nprocs_col
 *
 */

extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern "C" void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
extern "C" void blacs_gridexit_(int*);
extern "C" int numroc_(int*, int*, int*, int*, int*);

int main(int argc, char **argv) {
    int izero=0;
    int ione=1;
    int myrank_mpi, nprocs_mpi;
    MPI_Init( &argc, &argv);

    // Warmup MKL
    {
        double* A = (double*) calloc(256*256,sizeof(double));
        double* B = (double*) calloc(256*256,sizeof(double));
        double* C = (double*) calloc(256*256,sizeof(double));
        for(int i = 0; i < 256*256; i++) { A[i] = 1.0; B[i] = 1.0; C[i] = 1.0; }
        for(int i = 0; i < 10; i++) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 256, 256, 256, 1.0, A, 256, B, 256, 1.0, C, 256);
        }
        free(A); free(B); free(C);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    int n = 1000;       // (Global) Matrix size
    int nprow = 2;   // Number of row procs
    int npcol = 2;   // Number of column procs
    int nb = 256;      // (Global) Block size
    char uplo='L';   // Matrix is lower triangular
    char layout='R'; // Block cyclic, Row major processor mapping

    printf("Usage: ./test matrix_size block_size nprocs_row nprocs_col\n");

    if(argc > 1) {
        n = atoi(argv[1]);
    }
    if(argc > 2) {
        nb = atoi(argv[2]);
    }
    if(argc > 3) {
        nprow = atoi(argv[3]);
    }
    if(argc > 4) {
        npcol = atoi(argv[4]);
    }

    assert(nprow * npcol == nprocs_mpi);

    // Initialize BLACS
    int iam, nprocs;
    int zero = 0;
    int ictxt, myrow, mycol;
    blacs_pinfo_(&iam, &nprocs) ; // BLACS rank and world size
    blacs_get_(&zero, &zero, &ictxt ); // -> Create context
    blacs_gridinit_(&ictxt, &layout, &nprow, &npcol ); // Context -> Initialize the grid
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol ); // Context -> Context grid info (# procs row/col, current procs row/col)

    // Compute the size of the local matrices
    int mpA    = numroc_( &n, &nb, &myrow, &izero, &nprow ); // My proc -> row of local A
    int nqA    = numroc_( &n, &nb, &mycol, &izero, &npcol ); // My proc -> col of local A

    printf("Hi. Proc %d/%d for MPI, proc %d/%d for BLACS in position (%d,%d)/(%d,%d) with local matrix %dx%d, global matrix %d, block size %d\n",myrank_mpi,nprocs_mpi,iam,nprocs,myrow,mycol,nprow,npcol,mpA,nqA,n,nb);

    // Allocate and fill the matrices A and B
    // A[I,J] = (I == J ? 5*n : I+J)
    double *A;
    A = (double *)calloc(mpA*nqA,sizeof(double)) ;
    if (A==NULL){ printf("Error of memory allocation A on proc %dx%d\n",myrow,mycol); exit(0); }
    int k = 0;
    for (int j = 0; j < nqA; j++) { // local col
        int l_j = j / nb; // which block
        int x_j = j % nb; // where within that block
        int J   = (l_j * npcol + mycol) * nb + x_j; // global col
        for (int i = 0; i < mpA; i++) { // local row
            int l_i = i / nb; // which block
            int x_i = i % nb; // where within that block
            int I   = (l_i * nprow + myrow) * nb + x_i; // global row
            assert(I < n);
            assert(J < n);
            if(I == J) {
                A[k] = 1e5*n*n;
            } else {
                A[k] = I+J;
            }
            //printf("%d %d -> %d %d -> %f\n", i, j, I, J, A[k]);
            k++;
        }
    }

    // Create descriptor
    int descA[9];
    int info;
    int lddA = mpA > 1 ? mpA : 1;
    descinit_( descA,  &n, &n, &nb, &nb, &izero, &izero, &ictxt, &lddA, &info);
    if(info != 0) {
        printf("Error in descinit, info = %d\n", info);
    }

    // Run dpotrf and time
    printf("[%dx%d] Starting potrf\n", myrow, mycol);
    MPI_Barrier(MPI_COMM_WORLD);
    double MPIt1 = MPI_Wtime();
    pdpotrf_(&uplo, &n, A, &ione, &ione, descA, &info);
    if (info != 0) {
        printf("Error in potrf, info = %d\n", info);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double MPIt2 = MPI_Wtime();
    printf("[%dx%d] Done, time %e s.\n", myrow, mycol, MPIt2 - MPIt1);
    int ncores = -1;
    if(const char* env_p = std::getenv("MKL_NUM_THREADS")) ncores = atoi(env_p);
    printf(">>>>exp rank nranks ncores matrix_size block_size num_blocks total_time\n");
    printf("[%d]>>>>scalapack_pdpotrf %d %d %d %d %d %d %e\n",myrank_mpi,myrank_mpi,nprocs_mpi,ncores,n,nb,(n+nb-1)/nb,MPIt2-MPIt1);
    free(A);

    // Exit and finalize
    blacs_gridexit_(&ictxt);
    MPI_Finalize();
    return 0;
}
