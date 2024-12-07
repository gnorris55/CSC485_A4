
#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>

#include <GEMM.hpp>
#include "data_generator.h"

template <typename T>
void print_matrix(T* matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        std::cout << "| ";
        for (int j = 0; j < n; j++) {
            std::cout << matrix[i * m + j] << " ";

        }
        std::cout << "|" << std::endl;
    }
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */

int main()
{
    using namespace csc485b;

    std::size_t matrix_size = 4;

    std::vector<int> matrix_A = a1::generate_uniform<int>(16);
    std::vector<int> matrix_B = a1::generate_uniform<int>(16);

    // Create input

    
    std::vector<int> result = run_basic_matrix_multiplication<int>(matrix_A, matrix_B, matrix_size);
   

    print_matrix(matrix_A.data(), 4, 4);
    std::cout << "+" << std::endl;
    print_matrix(matrix_B.data(), 4, 4);
    std::cout << "=" << std::endl;
    print_matrix(result.data(), 4, 4);

     return EXIT_SUCCESS;
}