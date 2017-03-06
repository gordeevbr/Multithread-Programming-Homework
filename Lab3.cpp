#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <cmath>
#include <algorithm>
#include "mpi.h"

#ifdef _WIN32
#define WINPAUSE system("pause")
#endif

const std::string INFILE = "1.000.000.txt";
const std::string OUTFILE = "outfile.txt";

const int TAG_LEFT_ARRAY = 1001;
const int TAG_RIGHT_ARRAY = 1002;
const int TAG_LEFT_ARRAY_SIZE = 1003;
const int TAG_RIGHT_ARRAY_SIZE = 1004;
const int TAG_NEW_MEDIAN = 1005;
const int TAG_MERGE_SIZE = 1006;
const int TAG_MERGE_ARRAY = 1007;

const bool DEBUG_SWITCH = true;

void array_copy(int* array_source, int* array_target, int size) {
	for (int i = 0; i < size; i++) {
		array_target[i] = array_source[i];
	}
}

void read_array(int* array, std::ifstream &infile, int size) {
	for (int i = 0; i < size; i++) {
		infile >> array[i];
	}
}

void write_array(int* array, std::ofstream &outfile, int size) {
	for (int i = 0; i < size; i++) {
		outfile << array[i] << "\n";
	}
}

void swap(int* array, int i, int j) {
	int backup = array[i];
	array[i] = array[j];
	array[j] = backup;
}

void quicksort(int* array, int l, int r) {
	int x = array[l + (r - l) / 2];
	int i = l;
	int j = r;

	while (i < j) {
		while (array[i] < x) {
			i++;
		}
		
		while (array[j] > x) {
			j--;
		}

		if (i <= j) {
			swap(array, i, j);
			i++;
			j--;
		}
	}

	if (i < r) {
		quicksort(array, i, r);
	}

	if (l < j) {
		quicksort(array, l, j);
	}
}

bool correctness_check(int* array, int size) {
	for (int i = 0; i < size - 1; i++) {
		if (array[i + 1] < array[i]) {
			return false;
		}
	}
	return true;
}

void merge(int* left_array, int* left_array_size,
	int* right_array, int* right_array_size, int* buffer_array) {
	int buffer_array_size = 0;
	int i = 0;
	int j = 0;
	while (i < *left_array_size && j < *right_array_size) {
		if (left_array[i] < right_array[j]) {
			buffer_array[buffer_array_size] = left_array[i];
			i++;
			buffer_array_size++;
		} else {
			buffer_array[buffer_array_size] = right_array[j];
			j++;
			buffer_array_size++;
		}
	}
	while (i < *left_array_size) {
		buffer_array[buffer_array_size] = left_array[i];
		i++;
		buffer_array_size++;
	}
	while (j < *right_array_size) {
		buffer_array[buffer_array_size] = right_array[j];
		j++;
		buffer_array_size++;
	}
	std::copy(buffer_array, buffer_array + buffer_array_size, left_array);
	*left_array_size = buffer_array_size;
	*right_array_size = 0;
}

//Sorts right and left lists so that all values larger than median are in the right one
//and all values that are less than median are in the left one
void sort_by_median(int median, int* left_array, int* left_array_size, 
	int* right_array, int* right_array_size) {
	int* found = std::lower_bound(left_array, left_array + *left_array_size, median);
	int i = found - left_array + 1;
	int j = 0;
	while (i < *left_array_size) {
		right_array[j] = left_array[i];
		(*right_array_size)++;
		i++;
		j++;
	}
	*left_array_size = *left_array_size - *right_array_size;
}

int get_partner_process(int rank, int size, int depth) {
	int bit = log2(size) + 0.5;
	bit = bit - depth - 1;
	bit = pow(2, bit) + 0.5;
	return rank ^ bit;
}

void get_partner_group(int rank, int size, int depth,
	int *group_start, int* group_end) {
	int group_size = ((double)size / pow(2, depth)) + 0.5;
	int groups = size / group_size;
	for (int i = 0; i < groups; i++) {
		int start_index = i * group_size;
		int end_index = (i + 1) * group_size - 1;
		if (rank >= start_index && rank <= end_index) {
			*group_start = start_index;
			*group_end = end_index;
			break;
		}
	}
}

void hyperquicksort_iteration(int depth, int rank, int size, int array_size, int median,
		int* left_array, int *left_array_size, int* right_array, int *right_array_size,
		int* right_array_buffer) {
	//Produce 'left' and 'right' lists based on current median value
	sort_by_median(median, left_array, left_array_size, right_array, right_array_size);

	//Find partner and share 'left' and 'right' with it
	//Hypothetical deadlock?
	int partner = get_partner_process(rank, size, depth);
	if (partner > rank) {
		MPI_Send(left_array_size, 1, MPI_INT, partner, TAG_LEFT_ARRAY_SIZE, MPI_COMM_WORLD);
		MPI_Send(left_array, *left_array_size, MPI_INT, partner, TAG_LEFT_ARRAY, MPI_COMM_WORLD);
		int recieve_size;
		MPI_Recv(&recieve_size, 1, MPI_INT, partner, TAG_RIGHT_ARRAY_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(left_array, recieve_size, MPI_INT, partner, TAG_RIGHT_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		*left_array_size = recieve_size;
	} else {
		int recieve_size;
		MPI_Recv(&recieve_size, 1, MPI_INT, partner, TAG_LEFT_ARRAY_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(right_array_buffer, recieve_size, MPI_INT, partner, TAG_LEFT_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(right_array_size, 1, MPI_INT, partner, TAG_RIGHT_ARRAY_SIZE, MPI_COMM_WORLD);
		MPI_Send(right_array, *right_array_size, MPI_INT, partner, TAG_RIGHT_ARRAY, MPI_COMM_WORLD);
		std::copy(right_array_buffer, right_array_buffer + recieve_size, right_array);
		*right_array_size = recieve_size;
	}

	//Merge two new lists into one
	merge(left_array, left_array_size, right_array, right_array_size, right_array_buffer);

	//Find all group processes and send/recieve median
	int group_start = 0;
	int group_end = 0;
	get_partner_group(rank, size, depth, &group_start, &group_end);
	if (rank == group_start) {
		median = left_array[*left_array_size / 2];
		for (int i = group_start + 1; i <= group_end; i++) {
			MPI_Send(&median, 1, MPI_INT, i, TAG_NEW_MEDIAN, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(&median, 1, MPI_INT, group_start, TAG_NEW_MEDIAN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	depth++;

	//Return if more than log2(threadnum) iterations have passed
	if ((double(depth)) >= log2(size)) {
		return;
	}

	hyperquicksort_iteration(depth, rank, size, array_size, median, left_array,
		left_array_size, right_array, right_array_size, right_array_buffer);
}

void multithread_parse(int* array, int* array_res, int size, int array_size) {
	//Calculate chunk data and scatter it
	int processCount = (int)(ceil((double)array_size / (double)size) + 0.5);
	int processed = 0;
	int *sendSizes = (int *)malloc(size * sizeof(int));
	int *sendOffsets = (int *)malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		sendOffsets[i] = processed;
		int processedThisThread = min(processed + processCount, array_size);
		sendSizes[i] = processedThisThread - processed;
		processed = processedThisThread;
	}

	int mySize;
	MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(sendSizes, 1, MPI_INT, &mySize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Prepare left and right arrays
	int *left_array = (int *)malloc(array_size * sizeof(int));
	int *right_array = (int *)malloc(array_size * sizeof(int));
	int *right_array_buffer = (int *)malloc(array_size * sizeof(int));

	//Scatter chunks
	MPI_Scatterv(array, sendSizes, sendOffsets, MPI_INT, left_array, mySize, MPI_INT, 0, MPI_COMM_WORLD);

	//Sort local chunk
	quicksort(left_array, 0, mySize - 1);

	//Select the median and broadcast it
	int median = left_array[mySize / 2];
	MPI_Bcast(&median, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Iterate
	int* left_array_size;
	int* right_array_size;
	int right_array_size_val = 0;
	int merge_buffer_size_val = 0;
	left_array_size = &mySize;
	right_array_size = &right_array_size_val;
	int* merge_buffer_size = &merge_buffer_size_val;
	hyperquicksort_iteration(0, 0, size, array_size, median, left_array, 
		left_array_size, right_array, right_array_size, right_array_buffer);

	//Merge
	std::copy(left_array, left_array + *left_array_size, array_res + (array_size - *left_array_size));
	int array_res_size = *left_array_size;
	for (int i = 1; i < size; i++) {
		int recieve_size;
		MPI_Recv(&recieve_size, 1, MPI_INT, i, TAG_MERGE_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(array_res + (array_size - array_res_size - recieve_size), recieve_size, MPI_INT, i, TAG_MERGE_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		array_res_size += recieve_size;
	}

	//Delete leftover buffers
	free(sendSizes);
	free(sendOffsets);
	free(left_array);
	free(right_array);
	free(right_array_buffer);
}

void child_thread_program(int rank, int size) {
	//Recieve chunk data from master
	int mySize;
	int array_size;
	MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(NULL, 0, MPI_INT, &mySize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Prepare left and right arrays
	int *left_array = (int *)malloc(array_size * sizeof(int));
	int *right_array = (int *)malloc(array_size * sizeof(int));
	int *right_array_buffer = (int *)malloc(array_size * sizeof(int));

	//Recieve chunk from master
	MPI_Scatterv(NULL, 0, 0, MPI_INT, left_array, mySize, MPI_INT, 0, MPI_COMM_WORLD);

	//Sort local chunk
	quicksort(left_array, 0, mySize - 1);

	//Get median
	int median = 0;
	MPI_Bcast(&median, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Iterate
	int* left_array_size;
	int* right_array_size;
	int right_array_size_val = 0;
	int merge_buffer_size_val = 0;
	left_array_size = &mySize;
	right_array_size = &right_array_size_val;
	int* merge_buffer_size = &merge_buffer_size_val;
	hyperquicksort_iteration(0, rank, size, array_size, median, left_array, 
		left_array_size, right_array, right_array_size, right_array_buffer);

	//Send result data to master for merging
	MPI_Send(left_array_size, 1, MPI_INT, 0, TAG_MERGE_SIZE, MPI_COMM_WORLD);
	MPI_Send(left_array, *left_array_size, MPI_INT, 0, TAG_MERGE_ARRAY, MPI_COMM_WORLD);

	//Delete leftover buffers
	free(left_array);
	free(right_array);
	free(right_array_buffer);
}

void master_thread_program(int size) {
	std::ifstream infile(INFILE);
	std::ofstream outfile(OUTFILE);

	int array_size;
	infile >> array_size;

	int *array = (int *)malloc(array_size * sizeof(int *));
	int *array_check = (int *)malloc(array_size * sizeof(int *));
	int *array_res = (int *)malloc(array_size * sizeof(int *));

	read_array(array, infile, array_size);
	array_copy(array, array_check, array_size);

	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;

	QueryPerformanceCounter(&StartingTime);
	quicksort(array_check, 0, array_size-1);
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	std::cout << "Single thread calc time:" << ElapsedMicroseconds.QuadPart << std::endl;

	bool resultCorrect = correctness_check(array_check, array_size);
	std::cout << "Single thread result correct?: " << (resultCorrect ? "true" : "false") << std::endl;

	QueryPerformanceCounter(&StartingTime);
	multithread_parse(array, array_res, size, array_size);
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	std::cout << "Multiple thread calc time:" << ElapsedMicroseconds.QuadPart << std::endl;

	resultCorrect = correctness_check(array_res, array_size);
	std::cout << "Multiple thread result correct?: " << (resultCorrect ? "true" : "false") << std::endl;

	write_array(array_res, outfile, array_size);

	free(array);
	free(array_check);
	free(array_res);

	infile.close();
	outfile.flush();
	outfile.close();
}

bool size_ok(int size) {
	double res = log2(size);
	double intpart;
	double modf = std::modf(res, &intpart);
	return modf == 0.0 && intpart >= 1;
}

int main(int argc, char* argv[]) {
	MPI_Init(NULL, NULL);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		if (size_ok(size)) {
			master_thread_program(size);
		} else {
			std::cout << "Thread number must be a power of two and larger than or equal to 1." << std::endl;
		}
	}
	else {
		if (size_ok(size)) {
			child_thread_program(rank, size);
		}
	}

	MPI_Finalize();

	if (rank == 0) {
		WINPAUSE;
	}

	return 0;
}


