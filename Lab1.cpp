#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <chrono>
#include <string>

void calc(int** &matrix1, int m1, int n1,
          int** &matrix2, int m2, int n2,
          int** &res_matrix, int threads) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 10000)
    for(int i = 0; i < n1 * m2; i++){
        int n = i / n1;
        int m = i % n1;
        int product = 1;
        for (int j = 0; j < n2; j++) {
            product += matrix1[n][j] * matrix2[j][m];
        }
        res_matrix[n][m] = product;
    };
}

void init_matrix(int** &matrix, int m, int n) {
    matrix = new int*[n];
    for(int i = 0; i < n; i++) {
        matrix[i] = new int[m];
    }
}

void delete_matrix(int** &matrix, int n) {
    for(int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void print_result(int** &matrix, std::string filename) {
    std::ofstream outfile(filename);
    outfile.flush();
    outfile.close();
}

int main() {
    int TYPES_START = 4;
    int TYPES_MAX = 4;
    int THREADS_MAX = 10;
    int THREADS_START = 1;

    for (int i = TYPES_START; i <= TYPES_MAX; i++) {
        std::ostringstream is;
        is << i;
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

        if (m1 != n2) {
            std::cout << "Matrix dimensions don't match.";
            break;
        }

        int** matrix1;
        int** matrix2;
        int** res_matrix;
        init_matrix(matrix1, m1, n1);
        init_matrix(matrix2, m2, n2);
        init_matrix(res_matrix, m2, n1);

        for (int j = THREADS_START; j <= THREADS_MAX; j++) {
            auto start = std::chrono::system_clock::now();
            calc(matrix1, m1, n1, matrix2, m2, n2, res_matrix, j);
            auto end = std::chrono::system_clock::now();
            std::cout << "For " << j << " threads and (" << m1 << ";" << n1 << "), (" << m2 << ";" << n2 <<
                      ") matrices result time duration is " <<
                      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() <<
                      "ns, scheduling type is " << "dynamic" << std::endl;
        }

        print_result(res_matrix, std::string("out_").append(is.str()).append(".txt"));

        delete_matrix(matrix1, n1);
        delete_matrix(matrix2, n2);
        delete_matrix(res_matrix, n1);

        infile1.close();
        infile2.close();
    }

    return 0;
}