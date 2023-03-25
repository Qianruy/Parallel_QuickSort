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

// Recursive function
void quicksort(int *arr, int subsize, int n, MPI_Comm comm) {

    // Get the number and rank of processes
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // Serial sorting for p = 1
    if (p == 1) {
        std::sort(arr, arr + subsize);
        return;
    }

    int proc, pivot;

    // random number generator to generate
    if (rank == 0){ proc = rand() % p; }
    MPI_Bcast(&proc, 1, MPI_INT, 0, comm);
    if (rank == proc) {
        pivot = arr[rand() % (subsize - 1)];
        printf("pivot: %d\n", pivot);
    }
    
    // Use MPI_Bcast function to broadcast pivot to all processors
    MPI_Bcast(&pivot, 1, MPI_INT, proc, comm);
    // Wait for all clusters to reach this point 
    MPI_Barrier(comm);
    
    // print the pivot on each process
    // printf("pivot for rank %d: %d\n", world_rank, pivot);

    // local partition
    int left_size = partition(arr, 0, subsize, pivot);
    int right_size = subsize - left_size;
    printf("rank %d after partition: ", rank);
    for (int i = 0; i < subsize; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n\n");

    // Wait for all clusters to reach this point 
    MPI_Barrier(comm);

    // preifx sum for left array size and right array size
    int prefix_left[p];
    int prefix_right[p];
    int tmp_state;
    MPI_Scan(&left_size, &tmp_state, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tmp_state, 1, MPI_INT, prefix_left, 1, MPI_INT, comm);
    printf("rank %d received left size!\n", rank);
    if (rank == 0) {
        printf("prefix sum of left size: ");
        for (int i = 0; i < p; i++) {
            printf("%d ", prefix_left[i]);
        }
        printf("\n\n");
    }

    MPI_Scan(&right_size, &tmp_state, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tmp_state, 1, MPI_INT, prefix_right, 1, MPI_INT, comm);
    printf("rank %d received right size!\n", rank);
    if (rank == 0) {
        printf("prefix sum of right size: ");
        for (int i = 0; i < p; i++) {
            printf("%d ", prefix_right[i]);
        }
        printf("\n\n");
    }
    
    // split communicator for left and right array
    int left_sum = prefix_left[p - 1];
    int right_sum = prefix_right[p - 1];
    int p_left = (int)lround((left_sum * p)/n);
    if (p_left == 0 && left_sum > 0) { p_left++; }
    else if (p_left == p && right_sum > 0) { p_left--; }
    int p_right = p - p_left;
    printf("p left: %d, p right: %d\n", p_left, p_right);
    int newLen;
    if (rank < p_left) {
        newLen = (rank < (left_sum % p_left)) ? (left_sum / p_left + 1) : (left_sum / p_left);
    } else {
        newLen = (rank < (right_sum % p_right)) ? (right_sum / p_right + 1) : (right_sum / p_right);
    }
    printf("newLen for rank %d: %d\n",rank, newLen);
    
}

int main(int argc, char *argv[]){
    MPI_Status status;
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int n, subsize;
    int *arr, *subarr;
    
    // Get the number and rank of processes
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // random generator seed
    srand(10);

    if (world_rank == 0){
        // If command line arguments is invalid: error
        // if (argc != 2){
        //     printf("Error: invalid input!\n");
        //     exit(0);
        // }
        // Obtain input value n
        n = 16;
        arr = new int(n);
        for (int i = 0; i < n; i++) {
            arr[i] =(int) rand() % 100;
        }
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");  
    }
    // Use MPI_Wtime to time the run-time of the program
    double start;
    if (world_rank == 0) {start = MPI_Wtime();}

    // Use MPI_Bcast function to broadcast n to all processors
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // calculate the size of the subarray and displacements
    int subsize_arr[world_size];
    int displs[world_size];
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

    quicksort(subarr, subsize, n, MPI_COMM_WORLD);
    
    
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