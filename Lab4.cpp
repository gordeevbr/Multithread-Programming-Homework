#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <climits>

static const int UNREACHABLE = INT_MAX;
static const std::string INFILE = "input.txt";
static const std::string OUTFILE = "output.txt";
static const int START_VERTEX = 0;
static const int THREADS = 1;
static const int GENERATE_SIZE = 1000;
static const bool GENERATE = false;

#ifdef _WIN32
#define WINPAUSE system("pause")
#endif

void init_matrix(int** &matrix, int m, int n) {
	matrix = new int*[n];
	for (int i = 0; i < n; i++) {
		matrix[i] = new int[m];
		for (int j = 0; j < m; j++) {
			matrix[i][j] = 0;
		}
	}
}

void delete_matrix(int** &matrix, int size) {
	for (int i = 0; i < size; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

int index_of_smallest_element(int* array, int size) {
	int index = 0;
	for (int i = 1; i < size; i++) {
		if (array[i] < array[index]) {
			index = i;
		}
	}
	return index;
}

void multhithread_run(int** weights, int start_vertex, int* res_vector, int size, 
		bool* visited, int* min_value, int* min_index) {

	omp_set_num_threads(THREADS);

	int next_vertex = start_vertex;
	int vertex_weight;
	int index_of_smallest;

	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		if (i != start_vertex) {
			res_vector[i] = UNREACHABLE;
		} else {
			res_vector[i] = 0;
		}
		visited[i] = false;
	}

	while (next_vertex >= 0) {
		visited[next_vertex] = true;
		for (int i = 0; i < THREADS; i++) {
			min_value[i] = UNREACHABLE;
			min_index[i] = -1;
		}

		#pragma omp parallel for schedule(static) private(vertex_weight)
		for (int vertex = 0; vertex < size; vertex++) {
			vertex_weight = weights[next_vertex][vertex];
			if (next_vertex == vertex 
					|| visited[vertex] 
					|| vertex_weight == UNREACHABLE) {
				continue;
			}
			if (res_vector[vertex] > res_vector[next_vertex] + vertex_weight) {
				res_vector[vertex] = res_vector[next_vertex] + vertex_weight;
			}
			if (res_vector[vertex] < min_value[omp_get_thread_num()]) {
				min_value[omp_get_thread_num()] = res_vector[vertex];
				min_index[omp_get_thread_num()] = vertex;
			}
		}
		index_of_smallest = index_of_smallest_element(min_value, THREADS);
		next_vertex = min_index[index_of_smallest];
	}
}

void generate_matrix() {
	std::ofstream infile(INFILE);
	infile << GENERATE_SIZE << std::endl;
	for (int i = 0; i < GENERATE_SIZE; i++) {
		for (int j = 0; j < GENERATE_SIZE; j++) {
			if (i == j) {
				infile << 0 << " ";
			}
			else if (rand() % 2 == 0) {
				infile << UNREACHABLE << " ";
			}
			else {
				infile << rand() % 100 + 1 << " ";
			}
		}
		infile << std::endl;
	}
	infile.flush();
	infile.close();
}

void run() {
	std::ifstream infile(INFILE);
	std::ofstream outfile(OUTFILE);
	int size; 
	int** matrix;
	int* distances;
	int* min_value;
	int* min_index;
	bool* visited;
	double start_time;
	double end_time;

	infile >> size;
	init_matrix(matrix, size, size);
	distances = (int *)malloc(size * sizeof(int));
	visited = (bool *)malloc(size * sizeof(bool));
	min_value = (int *)malloc(THREADS * sizeof(int));
	min_index = (int *)malloc(THREADS * sizeof(int));

	for (int n = 0; n < size; n++) {
		for (int m = 0; m < size; m++) {
			infile >> matrix[n][m];
		}
	}

	infile.close();


	start_time = omp_get_wtime();
	multhithread_run(matrix, START_VERTEX, distances, size, visited, min_value, min_index);
	end_time = omp_get_wtime();
	std::cout << "Multithread calc time: " << end_time - start_time << std::endl;

	outfile << size << std::endl;
	for (int i = 0; i < size; i++) {
		outfile << distances[i] << " ";
	}
	outfile.flush();
	outfile.close();

	free(distances);
	free(visited);
	free(min_index);
	free(min_value);
	delete_matrix(matrix, size);
}

int main(int argc, char* argv[]) {
	if (GENERATE) {
		generate_matrix();
	} else {
		run();
	}
	WINPAUSE;
	return 0;
}