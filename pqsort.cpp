#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <iomanip>

void send_recv_matrix(int* sCount, int* rCount, int* prefix_prev, int recvLen, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    
    int prefix_new[p];
    int tmp_state;

    // Prefix Sum for destination size
    MPI_Scan(&recvLen, &tmp_state, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tmp_state, 1, MPI_INT, prefix_new, 1, MPI_INT, comm);

    // Compute index of destination subarray for current rank
    int front = 0;
    int end = 0;
    if (rank != 0) {
        while (prefix_new[front] < prefix_prev[rank - 1]) {
            front++;
        }
    }

    end = front;
    while (prefix_new[end] < prefix_prev[rank]) {
        end++;
    }

    // SendCounts array for current rank
    for (int i = 0; i < front; i++) {
        sCount[i] = 0;
    }

    int len;
    if (rank == 0) {
        len = prefix_prev[rank];
    } else {
        len = prefix_prev[rank] - prefix_prev[rank - 1];
    }

    if (front == end) {
        sCount[front] = len;
    } else {
        if (rank == 0) {
            sCount[front] = prefix_new[front];
        } else {
            sCount[front] = prefix_new[front] - prefix_prev[rank - 1];
        }
        for (int i = front + 1; i < end; i++) {
            sCount[i] = prefix_new[i] - prefix_new[i - 1];
        }
        sCount[end] = prefix_prev[rank] - prefix_new[end - 1];
    }

    for (int i = end + 1; i < p; i++) {
        sCount[i] = 0;
    }

    // rCount Array based on sCount
    for (int i = 0; i < p; i++) {
        MPI_Gather(&sCount[i], 1, MPI_INT, rCount, 1, MPI_INT, i, comm);
    }

    return;
}

int partition(std::vector<int> &arr, int low, int high, int pivot) {
    int i = low - 1, j = high;
    while (true) {
        do { i++; } while (i < high && arr[i] < pivot );
        do { j--; } while (j >= low && arr[j] > pivot); 
        if (i >= j) {
            return j;
        }
        std::swap(arr[i], arr[j]); 
    } 
}

// Organize result array 
std::vector<int> collectResult(std::vector<int> &subVec, int subLen, MPI_Comm comm) {
    int rank;
    int nPro;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nPro);
    std::vector<int> returnValue;

    // MPI_Barrier(comm);

    if (rank == 0) {
        std::vector<int> Allsizes(nPro);
        MPI_Gather(&subLen, 1, MPI_INT, &Allsizes[0], 1, MPI_INT, 0, comm);
        int sum_of_elems = std::accumulate(Allsizes.begin(), Allsizes.end(), 0);
        returnValue.resize(sum_of_elems);
        // Calculate displacements
        int *displsmts;
        displsmts = new int[nPro];
        displsmts[0] = 0;
        for (int i = 1; i < nPro; i++) {
            displsmts[i] = displsmts[i-1] + Allsizes[i-1];
        }

        // gather all the data to the root (0)
        MPI_Gatherv(&subVec[0], subLen, MPI_INT, &returnValue[0], &Allsizes[0], &displsmts[0], MPI_INT, 0, comm);
    } else {
        MPI_Gather(&subLen, 1, MPI_INT, NULL, 1, MPI_INT, 0, comm);
        MPI_Gatherv(&subVec[0], subLen, MPI_INT,
                    NULL, NULL, NULL, MPI_INT, 0, comm);
    }   
    return returnValue; 
}


// Recursive function
int quicksort(std::vector<int> &arr, int subsize, int n, MPI_Comm comm) {
    // Get the number and rank of processes
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int commLen;
    MPI_Allreduce(&subsize, &commLen, 1, MPI_INT, MPI_SUM, comm);

    // num of processors > num of numbers
    if (p > commLen) {
        MPI_Comm newComm;
        MPI_Comm_split(comm, rank < commLen, rank, &newComm);  
        if (rank >= commLen) {
            MPI_Comm_free(&newComm);
            return 0;
        }
        int debug = quicksort(arr, subsize, n, newComm);
        MPI_Comm_free(&newComm);
        return debug;
    }

    // Serial sorting for p = 1
    if (p == 1) {
        std::sort(arr.begin(), arr.end());
        return subsize;
    }

    // random number generator to generate
    int proc, pivot;
    // pick a processor
    if (rank == 0) { proc = rand() % p; }
    MPI_Bcast(&proc, 1, MPI_INT, 0, comm);
    if (rank == proc) {
        pivot = arr[rand() % subsize];
    }
    
    // Use MPI_Bcast function to broadcast pivot to all processors
    MPI_Bcast(&pivot, 1, MPI_INT, proc, comm);

    // local partition
    int left_size = partition(arr, 0, subsize, pivot) + 1;
    int right_size = subsize - left_size;

    // Wait for all clusters to reach this point 
    MPI_Barrier(comm);

    // preifx sum for left array size and right array size
    int prefix_left[p];
    int prefix_right[p];
    int tmp_state;
    MPI_Scan(&left_size, &tmp_state, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tmp_state, 1, MPI_INT, prefix_left, 1, MPI_INT, comm);

    MPI_Scan(&right_size, &tmp_state, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tmp_state, 1, MPI_INT, prefix_right, 1, MPI_INT, comm);
    
    // split communicator for left and right array
    int left_num = prefix_left[p - 1];
    int right_num = prefix_right[p - 1];

    int p_left = (int)round((left_num * p)/ (left_num + right_num));
    if (p_left == 0 && left_num > 0) { p_left++; }
    else if (p_left == p && right_num > 0) { p_left--; }
    int p_right = p - p_left;

    int newLen;
    if (rank < p_left) {
        newLen = (rank < (left_num % p_left)) ? (left_num / p_left + 1) : (left_num / p_left);
    } else {
        newLen = ((rank - p_left) < (right_num % p_right)) ? (right_num / p_right + 1) : (right_num / p_right);
    }
    int sCount[p];
    int rCount[p];
    int small_sCount[p];
    int small_rCount[p];

    int newSmLen;
    int newLgLen;

    if (rank >= p_left) {
        newSmLen = 0;
        newLgLen = newLen;
    } else {
        newSmLen = newLen;
        newLgLen = 0;
    }

    send_recv_matrix(small_sCount, small_rCount, prefix_left, newSmLen, comm);
    send_recv_matrix(sCount, rCount, prefix_right, newLgLen, comm);

    // Combine send and receive counts for small and large cases
    for (int i = 0; i < p; i++) {
        sCount[i] += small_sCount[i];
        rCount[i] += small_rCount[i];
    }

    // Get displacements array
    int sDispl[p];
    std::fill_n(sDispl, p, 0); 
    int rDispl[p];
    std::fill_n(rDispl, p, 0); 
    for (int i = 1; i < p; i++) {
        sDispl[i] = sDispl[i - 1] + sCount[i - 1];
        rDispl[i] = rDispl[i - 1] + rCount[i - 1];
    }

    // Alltoall data transfer
    int tempData[newLen];
    MPI_Alltoallv(&arr[0], sCount, sDispl, MPI_INT, &tempData[0], rCount, rDispl, MPI_INT, comm);
    
    // deep copy
    arr.resize(newLen);
    for (int i = 0; i < newLen; i++) {
        arr[i] = tempData[i];
    }

    // Split communicators
    MPI_Comm newComm;
    MPI_Comm_split(comm, (rank < p_left), rank, &newComm);

    // recursion
    int l = quicksort(arr, newLen, left_num + right_num, newComm);

    MPI_Comm_free(&newComm);

    return l;
}

int main(int argc, char *argv[]){
    MPI_Status status;
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int n, subsize;
    int *arr;
    
    // Get the number and rank of processes
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // random generator seed
    srand(5);

    if (world_rank == 0){
        // If command line arguments is invalid: error
        if (argc != 3){
            printf("Error: invalid input!\n");
            exit(0);
        }
        std::ifstream inputF;
        inputF.open(argv[1]);
        if (inputF.fail())   //checking whether the file is open
        {
            std::cout << "Input file failed to open."<< std::endl;
            exit(0);
        }
        std::string temp;
        if (!getline(inputF, temp))
        {
            std::cout << "Invalid n value in input file.\n"<< std::endl;
            exit(0);
        }
        n = stoi(temp);
        if (!getline(inputF, temp))
        {
            std::cout << "Invalid numbers in input file.\n"<< std::endl;
            exit(0);
        }
        inputF.close();
        std::stringstream ss(temp);
        std::vector<int> numbers;
        int x;
        while (ss >> x) {
            numbers.push_back(x);
        }
        arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = numbers[i];
        }

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

    std::vector<int> subarr(subsize);

    MPI_Scatterv(arr, subsize_arr, displs, MPI_INT, &subarr[0], subsize_arr[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Sort recursively
    int newL = quicksort(subarr, subsize, n, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double totaltime;
    if (world_rank == 0){totaltime = MPI_Wtime() - start;}  
    std::vector<int> sortedArray;
    sortedArray = collectResult(subarr, newL, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) { 
      int allSize = n;
      std::ofstream myfile;
      myfile.open(argv[2]);
      for (int i = 0; i < allSize - 1; i++) {
           myfile << sortedArray[i];
           myfile << " ";
      } 
      myfile << sortedArray[allSize - 1];
      myfile << "\n";
      myfile << std::fixed;
      myfile << std::setprecision(6);
      myfile <<  totaltime <<std::endl;
      myfile.close();
      printf("Output created\n");
    }

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
    return 0;
}
