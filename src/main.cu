
#include <cstddef>  // std::size_t type
#include <iostream> // std::cout, std::endl
#include <vector>

#include "algorithm_choices.h"
#include "data_generator.h"
#include "data_types.h"
<<<<<<< Updated upstream
#include "cuda_common.h"
=======


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
 * Runs timing tests on a CUDA graph implementation.
 * Consists of independently constructing the graph and then
 * modifying it to its two-hop neighbourhood.
 */
template < typename DeviceGraph >
void run(DeviceGraph g, csc485b::a2::edge_t const* d_edges, std::size_t m)
{
    
    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    // this code doesn't work yet!
    csc485b::a2::gpu::build_graph << < 1, 1 >> > (g, d_edges, m);

    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();


    // neither does this!   
    unsigned int tiling_size = 2;
    unsigned int matrix_size = sqrt(g.n);
    unsigned int num_block = matrix_size / tiling_size;
    csc485b::a2::gpu::two_hop_reachability <<< {num_block, num_block}, { tiling_size, tiling_size } >>> (g);

    
    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    
    std::cout << "Build time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
        << " us"
        << std::endl;
    
    std::cout << "Reachability time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
        << " us"
        << std::endl;
    
}

/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m)
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t* d_matrix;
    cudaMalloc((void**)&d_matrix, sizeof(a2::node_t) * n * n);
    a2::DenseGraph d_dg{ n, d_matrix };

    run(d_dg, d_edges, m);

    // check output?
    std::vector< a2::node_t > host_matrix(d_dg.matrix_size());
    a2::DenseGraph dg{ n, host_matrix.data() };
    cudaMemcpy(dg.adjacencyMatrix, d_dg.adjacencyMatrix, sizeof(a2::node_t) * d_dg.matrix_size(), cudaMemcpyDeviceToHost);
    std::copy(host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));

    // clean up
    cudaFree(d_matrix);
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m)
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t* d_offsets, * d_neighbours;
    cudaMalloc((void**)&d_offsets, sizeof(a2::node_t) * n);
    cudaMalloc((void**)&d_neighbours, sizeof(a2::node_t) * m);
    a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    run(d_sg, d_edges, m);

    // clean up
    cudaFree(d_neighbours);
    cudaFree(d_offsets);
}
>>>>>>> Stashed changes

int main()
{
    std::size_t const n = 32*32;
    std::size_t const switch_at = 3 * (n >> 2);

<<<<<<< Updated upstream
    auto data = csc485b::a1::generate_uniform< element_t >(n);
    csc485b::a1::cpu::run_cpu_baseline(data, switch_at, n);
    csc485b::a1::gpu::run_gpu_soln(data, switch_at, n);
=======
    // Create input
    std::size_t constexpr n = 4;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph(n, n * expected_degree);
    std::size_t const m = graph.size();

    // lazily echo out input graph
    for (auto const& e : graph)
    {
        std::cout << "(" << e.x << "," << e.y << ") ";
    }

    // allocate and memcpy input to device
    a2::edge_t* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(a2::edge_t) * m);
    cudaMemcpyAsync(d_edges, graph.data(), sizeof(a2::edge_t) * m, cudaMemcpyHostToDevice);

    // run your code!
    run_dense(d_edges, n, m);
   // run_sparse(d_edges, n, m);
>>>>>>> Stashed changes

    std::cout << "hello world!" << std::endl;
    return EXIT_SUCCESS;
}