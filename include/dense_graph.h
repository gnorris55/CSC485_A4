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
                void build_graph(DenseGraph g, std::size_t m)
            {
                // IMPLEMENT ME!
                return;
            }

            __device__
                void matrix_mult(int *adj_mat, int n) {


                
                const int tiling_size = 2;

                __shared__ int vert_smem[tiling_size][tiling_size];
                __shared__ int horizontal_smem[tiling_size][tiling_size];
                __syncthreads();


                int row = blockIdx.y * tiling_size + threadIdx.y;
                int col = blockIdx.x * tiling_size + threadIdx.x;

                if (row == col) {
                    adj_mat[row * n + col] = 0;
                    return;
                }


                float temp = 0;

                // width represents the column length of the matrix
                for (int i = 0; i < n / tiling_size; i++) {

                    // all values in A and B that are in the block will be loaded into shared memory
                    // want to syncronize threads so that all values are loaded into shared memory before we proceed
                    horizontal_smem[threadIdx.y][threadIdx.x] = adj_mat[row * n + (i * tiling_size + threadIdx.x)];
                    vert_smem[threadIdx.y][threadIdx.x] = adj_mat[(i * tiling_size + threadIdx.y) * n + col];
                    __syncthreads();

                    // will calculate the dot product for twp of the values
                    for (int k = 0; k < tiling_size; k++) {
                        temp += horizontal_smem[threadIdx.y][k] * vert_smem[k][threadIdx.x];
                        __syncthreads();
                    }
                }
                
                adj_mat[row * n + col] = temp;
            }

            /**
              * Repopulates the adjacency matrix as a new graph that represents
              * the two-hop neighbourhood of input graph g
              */
            __global__
                void two_hop_reachability(DenseGraph g)
            {

                matrix_mult(g.adjacencyMatrix, g.n);
                // IMPLEMENT ME!
                // square adjacencyMatrix
                // then remove the diagonal and clamp values back to [0,1]
                return;
            }

        } // namespace gpu
    } // namespace a2
} // namespace csc485b