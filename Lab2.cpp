#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <cmath>
#include "mpi.h"

#ifdef _WIN32
#define WINPAUSE system("pause")
#endif

const int M_SIZE = 4;
const int ITERATIONS_CAP = 100;
const double EPSILON = 1E-6;
const double EPSILON_CORRECTNESS = 1E-6;

const int TAG_MATRIX_SIZE = 1004;
const int TAG_NEXT_PAYLOAD_STATUS = 1005;
const int TAG_CONVERGES = 1006;

const int NEXT_PAYLOAD_STATUS_SENDING = 1;
const int NEXT_PAYLOAD_STATUS_BREAK = 0;

const int CONVERGES_TRUE = 1;
const int CONVERGES_FALSE = 0;

void init_matrix(double** &matrix, int m, int n) {
	matrix = new double*[n];
	for (int i = 0; i < n; i++) {
		matrix[i] = new double[m];
		for (int j = 0; j < m; j++) {
			matrix[i][j] = 0;
		}
	}
}

void init_matrix_contiguous(double** &matrix, int m, int n) {
	double* data = (double *)malloc(n*m * sizeof(double));
	if (data == NULL) {
		std::cout << "Malloc failed." << std::endl;
	}
	matrix = (double **)malloc(n * sizeof(double*));
	for (int i = 0; i < n; i++) {
		matrix[i] = &(data[m*i]);
		for (int j = 0; j < m; j++) {
			matrix[i][j] = 0;
		}
	}
}

void delete_array(double* array) {
	free(array);
}

void delete_array_int(int* array) {
	free(array);
}

void delete_matrix(double** &matrix, int n) {
	for (int i = 0; i < n; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

void delete_matrix_contiguous(double** &matrix) {
	//free(matrix[0]);
	//free(matrix);
}

void read_array(double* array, std::ifstream &infile, int n) {
	for (int i = 0; i < n; i++) {
		infile >> array[i];
	}
}

void read_matrix_AB(double** &A, double* B, std::ifstream &infile, int m, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (j != m - 1) {
				infile >> A[i][j];
			} else {
				infile >> B[i];
			}
		}
	}
}

void calc_normally(double** &A, int m,
	double* B, double* X,
	double* res_matrix) {
	for (int i = 0; i < m; i++) {
		res_matrix[i] = X[i];
	}
	double nonDiagProduct = 0;
	int iterations = 0;
	bool converges = false;
	do {
		converges = true;
		for (int i = 0; i < m; i++) {
			nonDiagProduct = 0;
			for (int j = 0; j < m; j++) {
				if (j != i) {
					nonDiagProduct += A[i][j] * res_matrix[j];
				}
			}
			res_matrix[i] = (B[i] - nonDiagProduct) / A[i][i];
			converges = converges && abs(B[i] - res_matrix[i]) <= EPSILON;
		}
		iterations++;
	} while (!converges && iterations < ITERATIONS_CAP);
}

bool correctnessCheck(double** &A, int m,
	double* B, double* X) {
	for (int i = 0; i < m; i++) {
		double product = 0;
		for (int j = 0; j < m; j++) {
			product += A[i][j] * X[j];
		}
		if (abs(B[i] - product) > EPSILON_CORRECTNESS) {
			return false;
		}
	}
	return true;
}

bool sufficientToConverge(double** &A, int m) {
	for (int i = 0; i < m; i++) {
		double product = 0;
		for (int j = 0; j < m; j++) {
			if (i != j) {
				product += abs(A[i][j]);
			}
		}
		if (abs(A[i][i]) <= product) {
			return false;
		}
	}
	return true;
}

bool calcPartially(double** &A, int m,
	double* B, double* X, double* res,
	int offset, int count) {
	bool converges = true;
	double nonDiagProduct;
	int trueIndex;
	for (int index = 0; index < count; index++) {
		trueIndex = offset + index;
		nonDiagProduct = 0;
		for (int j = 0; j < m; j++) {
			if (j != trueIndex) {
				nonDiagProduct += A[trueIndex][j] * X[j];
			}
		}
		res[index] = (B[trueIndex] - nonDiagProduct) / A[trueIndex][trueIndex];
		converges = converges && abs(res[index] - X[trueIndex]) <= EPSILON * EPSILON;
	}
	return converges;
}

void multithreadParse(double** &A, int m,
	double* B, double* X, double* res_matrix, int size) {

	//Broadcast common data
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(A[0][0]), m*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Calculate chunk data and scatter it
	int processCount = (int)(ceil((double)m / (double)size) + 0.5);
	int processed = 0;
	int *sendSizes = (int *)malloc(size * sizeof(int));
	int *sendOffsets = (int *)malloc(size * sizeof(int));
	if (sendSizes == NULL || sendOffsets == NULL) {
		std::cout << "Malloc failed." << std::endl;
	}
	for (int i = 0; i < size; i++) {
		sendOffsets[i] = processed;
		int processedThisThread = min(processed + processCount, m);
		sendSizes[i] = processedThisThread - processed;
		processed = processedThisThread;
	}

	int myOffset = 0;
	int mySize = 0;
	MPI_Scatter(sendSizes, 1, MPI_INT, &mySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(sendOffsets, 1, MPI_INT, &myOffset, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Prepare for iterating
	for (int i = 0; i < m; i++) {
		res_matrix[i] = X[i];
	}
	bool converges = false;
	int iterations = 0;

	double* myChunk = (double *)malloc(mySize * sizeof(double*));
	if (myChunk == NULL) {
		std::cout << "Malloc failed." << std::endl;
	}

	int next = NEXT_PAYLOAD_STATUS_SENDING;

	do {

		//Tell other threads they should gather
		MPI_Bcast(&next, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(res_matrix, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		//Calc own part
		converges = calcPartially(A, m, B, res_matrix, myChunk, myOffset, mySize);

		//Recieve other parts
		MPI_Gatherv(myChunk, mySize, MPI_DOUBLE, res_matrix, sendSizes, sendOffsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for (int i = 0; i < size - 1; i++) {
			int recvConverges;
			MPI_Recv(&recvConverges, 1, MPI_INT, i + 1, TAG_CONVERGES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			converges = converges && recvConverges == CONVERGES_TRUE;
		}

		iterations++;
	} while (!converges && iterations < ITERATIONS_CAP);

	//Tell other threads you're done
	next = NEXT_PAYLOAD_STATUS_BREAK;
	MPI_Bcast(&next, 1, MPI_INT, 0, MPI_COMM_WORLD);

	delete_array_int(sendSizes);
	delete_array_int(sendOffsets);
}

void childThreadProgram(int rank, int size) {

	//Prepare common data
	int m = 0;
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

	double** A;
	double* B = (double *)malloc(m * sizeof(double*));
	double* X = (double *)malloc(m * sizeof(double*));
	init_matrix_contiguous(A, m, m);
	if (A == NULL || X == NULL || B == NULL) {
		std::cout << "Malloc failed." << std::endl;
	}

	MPI_Bcast(&(A[0][0]), m*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Recieve personal chunk info and prepare before iterating
	int mySize;
	int myOffset;
	MPI_Scatter(NULL, 1, MPI_INT, &mySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(NULL, 1, MPI_INT, &myOffset, 1, MPI_INT, 0, MPI_COMM_WORLD);

	double* myChunk = (double *)malloc(mySize * sizeof(double*));
	if (myChunk == NULL) {
		std::cout << "Malloc failed." << std::endl;
	}

	int continueIterating;
	MPI_Bcast(&continueIterating, 1, MPI_INT, 0, MPI_COMM_WORLD);

	bool converges;

	int convergesInt;
	//Start iterating
	while (continueIterating == NEXT_PAYLOAD_STATUS_SENDING) {

		//Update X matrix
		MPI_Bcast(X, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		//Calc own part
		converges = calcPartially(A, m, B, X, myChunk, myOffset, mySize);

		//Accumulate parts on master thread
		MPI_Gatherv(myChunk, mySize, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		convergesInt = (converges ? CONVERGES_TRUE : CONVERGES_FALSE);
		MPI_Send(&convergesInt, 1, MPI_INT, 0, TAG_CONVERGES, MPI_COMM_WORLD);

		//See if there's a next iteration
		MPI_Bcast(&continueIterating, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}


	//Cleanup
	delete_matrix_contiguous(A);
	delete_array(B);
	delete_array(X);
	delete_array(myChunk);
}

void masterThreadProgram(int size) {
	std::ostringstream is;
	is << M_SIZE;
	std::ifstream infile1(
		std::string(is.str())
		.append("_1.txt")
	), infile2(
		std::string(is.str())
		.append("_2.txt")
	);

	int m1, n1, m2, n2;
	infile1 >> m1 >> n1;
	infile2 >> m2 >> n2;

	std::cout << "m1: " << m1 << std::endl
		<< "n1: " << n1 << std::endl
		<< "m2: " << m2 << std::endl
		<< "n2: " << n2 << std::endl;

	if (m1 != n1 + 1 || m2 != 1 || n1 != n2) {
		std::cout << "Matrix dimensions are wrong." << std::endl;
		return;
	}

	double **A;
	double *B = (double *)malloc(n2 * sizeof(double*));
	double *X = (double *)malloc(n2 * sizeof(double*));
	double *check_matrix = (double *)malloc(n2 * sizeof(double*));
	double *res_matrix = (double *)malloc(n2 * sizeof(double*));
	if (B == NULL || X == NULL || check_matrix == NULL || res_matrix == NULL) {
		std::cout << "malloc error." << std::endl;
	}
	init_matrix_contiguous(A, m1 - 1, n1);

	read_matrix_AB(A, B, infile1, m1, n1);
	read_array(X, infile2, n2);

	bool sufficientConditionMet = sufficientToConverge(A, n1);
	if (!sufficientConditionMet) {
		std::cout << "Matrix does not meet sufficient convergence requirement, thus it might not converge." << std::endl;
	}

	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;

	QueryPerformanceFrequency(&Frequency);

	QueryPerformanceCounter(&StartingTime);
	calc_normally(A, n1, B, X, check_matrix);
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	std::cout << "Single thread calc time:" << ElapsedMicroseconds.QuadPart << std::endl;

	bool resultCorrect = correctnessCheck(A, n1, B, check_matrix);
	std::cout << "Single thread result correct?: " << (resultCorrect ? "true" : "false") << std::endl;

	QueryPerformanceCounter(&StartingTime);
	multithreadParse(A, n1, B, X, res_matrix, size);
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	std::cout << "Multiple thread calc time:" << ElapsedMicroseconds.QuadPart << std::endl;

	for (int i = 0; i < n2; i++) {
		std::cout << res_matrix[i] << std::endl;
	}

	resultCorrect = correctnessCheck(A, n1, B, res_matrix);
	std::cout << "Multiple thread result correct?: " << (resultCorrect ? "true" : "false") << std::endl;

	delete_matrix_contiguous(A);
	delete_array(B);
	delete_array(X);
	delete_array(check_matrix);
	delete_array(res_matrix);

	infile1.close();
	infile2.close();
}

int main(int argc, char* argv[]) {
	MPI_Init(NULL, NULL);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		masterThreadProgram(size);
	} else {
		childThreadProgram(rank, size);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (rank == 0) {
		WINPAUSE;
	}

	return 0;
}
