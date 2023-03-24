#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 


int partition(int *arr, int low, int high, int pivot) {
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[j], arr[i]);
        }
    }
    return (i+1);
}

// int partition_processor(int p, int left_size, int right_size, MPI_Comm MPI_COMM_WORLD) {

// }

int main(int argc, char *argv[]){
    MPI_Status status;
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int n, subsize;
    int pivot_idx, pivot;
    int *arr, *subarr;
    
    // Get the number and rank of processes
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int subsize_arr[world_size];
    int displs[world_size];

    if (world_rank == 0){
        // If command line arguments is invalid: error
        // if (argc != 2){
        //     printf("Error: invalid input!\n");
        //     exit(0);
        // }
        // Obtain input value n
        n = 12;
        arr = new int(n);
        for (int i = 0; i < n; i++) {
            arr[i] =(int) rand() % 100;
        }
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
        // random number generator to generate
        srand(100);
        pivot_idx = rand() % (n - 1);
        pivot = arr[pivot_idx];
    }
    
    // Use MPI_Bcast function to broadcast n to all processors
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pivot, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Wait for all clusters to reach this point 
    MPI_Barrier(MPI_COMM_WORLD);
    // Use MPI_Wtime to time the run-time of the program
    double start;
    if (world_rank == 0) {start = MPI_Wtime();}

    // calculate the size of the subarray and displacements
    int prefix = 0;
    for (int i = 0; i < world_size; i++) {
        subsize = (i < (n % world_size)) ? (n / world_size + 1) : (n / world_size);
        subsize_arr[i] = subsize;
        displs[i] = prefix;
        prefix += subsize;
    }
    // integers are equally distributed to the processors
    // calculate the size of the subarray for each process
    subsize = (world_rank < (n % world_size)) ? (n / world_size + 1) : (n / world_size);
    printf("subsize for rank %d: %d\n", world_rank, subsize);
    subarr = new int(subsize);
    MPI_Scatterv(arr, subsize_arr, displs, MPI_INT, subarr, subsize_arr[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    // print the subarray on each process
    printf("rank %d received: ", world_rank);
    for (int i = 0; i < subsize; i++) {
        printf("%d ", subarr[i]);
    }
    printf("\n");
    // print the pivot on each process
    printf("pivot for rank %d: %d\n", world_rank, pivot);
    
    // local partition
    int left_size = partition(subarr, 0, subsize, pivot);
    printf("rank %d after partition: ", world_rank);
    for (int i = 0; i < subsize; i++) {
        printf("%d ", subarr[i]);
    }
    printf("\n\n");

    if (world_rank == 0){ double end = MPI_Wtime();}    

    // free the memory
    if (world_rank == 0) {
        free(arr);
    }
    free(subarr);
    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
    return 0;
}