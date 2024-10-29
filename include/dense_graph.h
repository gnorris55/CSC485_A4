#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
    namespace a2 {

        /**
         * A DenseGraph is optimised for a graph in which the number of edges
         * is close to n(n-1). It is represented using an adjacency matrix.
         */
        struct DenseGraph
        {
            std::size_t n; /**< Number of nodes in the graph. */
            node_t* adjacencyMatrix; /** Pointer to an n x n adj. matrix */

            /** Returns number of cells in the adjacency matrix. */
            __device__ __host__ __forceinline__
                std::size_t matrix_size() const { return n * n; }
        };


        namespace gpu {


            /**
             * Constructs a DenseGraph from an input edge list of m edges.
             *
             * @pre The pointers in DenseGraph g have already been allocated.
             */
            __global__
            void build_graph(DenseGraph g, edge_t const * edge_list, std::size_t m)
            {
                // Get thread ID
                const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

                if (thread_id < m) {

                    // Edges associate to DenseGraph via indices (Row, Column)
                    // Given edge (A, B), get flat index by formula: idx = n(A)+B
                    std::size_t graph_index = g.n * edge_list[thread_id].x + edge_list[thread_id].y;

                    // Set dense graph bit
                    g.adjacencyMatrix[graph_index] = 1;
                }
                return;
            }

            __device__
                int get_min(int a, int b) {
                if (a > b)
                    return b;
                else
                    return a;

            }

            __device__ 
                void advance_matrix_mult(DenseGraph g) {

                unsigned int n = (int)g.n;
                __shared__ int results[32];


                int row = blockIdx.y;
                int col = blockIdx.x;


                if (threadIdx.x + threadIdx.y < n * n) {
                    
                    // do not want any self loops included
                    if (row == col) {
                        g.adjacencyMatrix[row * n + col] = 0;
                        return;
                    }


                    float temp = 0;
                    int hop_value = 0;

                    int vertical_value = g.adjacencyMatrix[(32 * threadIdx.y + threadIdx.x) * n + col];
                    int horizontal_value = g.adjacencyMatrix[(row * n) + 32 * threadIdx.y + threadIdx.x];
                    int value = vertical_value * horizontal_value;

                    __syncthreads();

                    hop_value = __any_sync(0xf, value > 0);

                    if (threadIdx.x == 0)
                        results[threadIdx.y] = hop_value;

                    __syncthreads();
                    int result = results[threadIdx.x];
                    g.adjacencyMatrix[blockIdx.y * n + blockIdx.x] = __any_sync(0xf, result > 0);

                }

               
                return;



            }
            __device__ 
                void normal_matrix_mult(DenseGraph g) {

                unsigned int n = (int)g.n;
                int tiling_size = get_min(32, (int)g.n);
                //const int tiling_size = 32;

                __shared__ int vert_smem[32][32];
                __shared__ int horizontal_smem[32][32];
                __syncthreads();


                int row = blockIdx.y * tiling_size + threadIdx.y;
                int col = blockIdx.x * tiling_size + threadIdx.x;

                if (row == col) {
                    g.adjacencyMatrix[row * n + col] = 0;
                    return;
                }


                float temp = 0;

                // width represents the column length of the matrix
                for (int i = 0; i < n / tiling_size; i++) {

                    // all values in A and B that are in the block will be loaded into shared memory
                    // want to syncronize threads so that all values are loaded into shared memory before we proceed
                    horizontal_smem[threadIdx.y][threadIdx.x] = g.adjacencyMatrix[row * n + (i * tiling_size + threadIdx.x)];
                    vert_smem[threadIdx.y][threadIdx.x] = g.adjacencyMatrix[(i * tiling_size + threadIdx.y) * n + col];
                    __syncthreads();


                    //replace this with warp level
                    // will calculate the dot product for twp of the values
                    for (int k = 0; k < tiling_size; k++) {
                       horizontal_smem[threadIdx.y][k] * vert_smem[k][threadIdx.x];
                        __syncthreads();
                    }
                    // the for loop above should not exist
                }
                if (temp > 0)
                    temp = 1;

                g.adjacencyMatrix[row * n + col] = temp;
                return;



            }
            __global__
                void two_hop_reachability(DenseGraph g)
            {
                advance_matrix_mult(g);
            }

        } // namespace gpu
    } // namespace a2
} // namespace csc485b