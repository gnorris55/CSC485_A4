

#include <mma.h>


__device__
int get_min(int a, int b) {
    if (a > b)
        return b;
    else
        return a;
}

template < typename T >
__global__
void matrix_mult(T *A, T *B, T *C, std::size_t n) {

       int tiling_size = get_min(32, n);

    __shared__ int vert_smem[32][32];
    __shared__ int horizontal_smem[32][32];
    __syncthreads();


    int row = blockIdx.y * tiling_size + threadIdx.y;
    int col = blockIdx.x * tiling_size + threadIdx.x;


    float temp = 0;

    // width represents the column length of the matrix
    for (int i = 0; i < n / tiling_size; i++) {

        // all values in A and B that are in the block will be loaded into shared memory
        // want to syncronize threads so that all values are loaded into shared memory before we proceed
        horizontal_smem[threadIdx.y][threadIdx.x] = A[row * n + (i * tiling_size + threadIdx.x)];
        vert_smem[threadIdx.y][threadIdx.x] = B[(i * tiling_size + threadIdx.y) * n + col];
        __syncthreads();


        //replace this with warp level
        // will calculate the dot product for twp of the values
        for (int k = 0; k < tiling_size; k++) {
            horizontal_smem[threadIdx.y][k] * vert_smem[k][threadIdx.x];
            __syncthreads();
        }
        // the for loop above should not exist
    }

    C[row * n + col] = temp;
    return;
}


template <typename T>
std::vector<T> run_basic_matrix_multiplication(std::vector<T> A, std::vector<T> B, std::size_t n) {
    
    T* d_A;
    T* d_B;
    T* d_C;

    std::vector<T> C(A.size(), -1);

    cudaMalloc((void**) &d_A, A.size() * sizeof(T));
    cudaMalloc((void**) &d_B, B.size() * sizeof(T));
    cudaMalloc((void**) &d_C, C.size() * sizeof(T));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), C.size() * sizeof(T), cudaMemcpyHostToDevice);


    unsigned int tiling_size = std::ceil(n / 32);
    unsigned int matrix_size = n;

    matrix_mult <<< {matrix_size, matrix_size}, { 32, tiling_size } >> > (d_A, d_B, d_C, n);

    cudaMemcpy(C.data(), d_C, sizeof(T) * C.size(), cudaMemcpyDeviceToHost);

    return C;
}
