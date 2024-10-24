#include "algorithm_choices.h"

#include <algorithm> // std::min_element()
#include <cassert>   // for assert()
#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

#include "cuda_common.h"

namespace csc485b {
    namespace a1 {
        namespace gpu {

            /**
             * The CPU baseline benefits from warm caches because the data was generated on
             * the CPU. Run the data through the GPU once with some arbitrary logic to
             * ensure that the GPU cache is warm too and the comparison is more fair.
             */
            __global__
                void warm_the_gpu(element_t* data, std::size_t invert_at_pos, std::size_t num_elements)
            {
                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

                // We know this will never be true, because of the data generator logic,
                // but I doubt that the compiler will figure it out. Thus every element
                // should be read, but none of them should be modified.
                if (th_id < num_elements && data[th_id] > num_elements * 100)
                {
                    ++data[th_id]; // should not be possible.
                }
            }


            template < typename T >
            __device__
                void swap(T& left, T& right)
            {
                T const temp = left;
                left = right;
                right = temp;
            }

            template < bool ASCENDING >
            __device__ __forceinline__
                bool should_swap(int my_th_id, int their_th_id, element_t my_val, element_t their_val, int group_size)
            {
                if (ASCENDING)
                    return ((their_th_id > my_th_id && !(group_size & my_th_id) && their_val < my_val)
                        || (their_th_id > my_th_id && (group_size & my_th_id) && their_val > my_val)
                        || (their_th_id < my_th_id && !(group_size & my_th_id) && their_val > my_val)
                        || (their_th_id < my_th_id && (group_size & my_th_id) && their_val < my_val));
                else
                    return ((their_th_id > my_th_id && !(group_size & my_th_id) && their_val > my_val)
                        || (their_th_id > my_th_id && (group_size & my_th_id) && their_val < my_val)
                        || (their_th_id < my_th_id && !(group_size & my_th_id) && their_val < my_val)
                        || (their_th_id < my_th_id && (group_size & my_th_id) && their_val > my_val));
            }

            namespace single_block {

                template < bool REVERSE >
                __device__
                    void bitonic_sort(element_t* data, std::size_t num_elements)
                {
                    int const th_id = threadIdx.x;

                    // create a reference to the index for which this thread is responsible so that we
                    // can refer to it more cleanly by name for now on.
                    auto my_val = data[th_id];

                    // macro-steps of bitonic sort double each iteration and determine the size of the
                    // block that should be sorted. We begin by sorting pairs and eventually sort the whole list
                    for (int group_size = 2; group_size <= num_elements; group_size = group_size << 1)
                    {
                        // On every micro-step, we merge sorted runs of half the size as before
                        for (int stride = group_size / 2; stride > 0; stride = stride >> 1)
                        {
                            // the thread with which this one will swap is determined by the stride,
                            // similar to a butterfly pattern. We get both the index with which to swap
                            // and a reference to that value in the input array
                            int paired_thread = th_id ^ stride;
                            element_t const paired_val = data[paired_thread];

                            if (should_swap< true >(th_id, paired_thread, my_val, paired_val, group_size))
                            {
                                data[th_id] = paired_val;
                                my_val = paired_val;
                            }
                            __syncthreads(); // need to ensure all swaps are complete before the next step
                        }
                    }

                    if (REVERSE)
                    {
                        // reverse sort in last quarter of array
                        // we do this efficiently by pairing up index i and n-i
                        // and having both overwrite each other's values.
                        // The math is easier if we work with the first quarter of the array
                        // and then just add the 3/4s offset to the address we want to write to.
                        int const reversal_size = num_elements >> 2;
                        int const reverse_at = 3 * reversal_size;
                        if (th_id >= reverse_at)
                        {
                            int const paired_location = num_elements - th_id - 1;
                            data[3 * reversal_size + paired_location] = my_val;
                        }
                    }
                }


                /**
                 * This solution uses a single block to perform a bitonic sort of up to 1024 elements.
                 * There is no use of shared memory or any clever tricks, just a minimisation of operations.
                 * Branch divergence is minimised by collapsing all the conditions together in `should_swap()`.
                 * Swaps and the final reversal are done collaboratively by pairs of threads to double throughput.
                 * Each thread is only concerned with what should be the value in the index for which it is responsible.
                 */
                __global__
                    void opposing_sort(element_t* data, std::size_t num_elements)
                {
                    assert("this solution assumes the input size is a power of two" && __popc(num_elements) == 1);
                    assert("this solution only works up to n=1024." && num_elements <= 1024);

                    extern __shared__ element_t smem[];

                    // Linear solution. No clever 2d grids and no multiple blocks. Just grab the x-dim thread id
                    // which is in the range [0, 1024).
                    int const th_id = threadIdx.x;

                    // ensure __syncthreads() is safe later by killing all threads
                    // that aren't doing something productive.
                    if (th_id >= num_elements) { return; }

                    // worked in shared memory instead
                    smem[th_id] = data[th_id];
                    __syncthreads();

                    bitonic_sort< true >(smem, num_elements);

                    // copy result back to input/output array
                    __syncthreads();
                    data[th_id] = smem[th_id];
                }

            } // namespace single_block



            namespace hierarchical {

                /**
                 * Perform a bitonic sort in registers using one warp (hence pass by ref).
                 * Two template instantiations control whether the sort is ascending or descending.
                 * Optionally, just run the last merge phase instead of the full sort.
                 */
                template < bool ASCENDING >
                __device__ __forceinline__
                    void bitonic_warp(element_t& my_val, bool merge_only = false)
                {
                    int const warp_size = blockDim.x;
                    int const lane_id = threadIdx.x;

                    assert("Single warp on x axis in launch config" && blockDim.x == warp_size);

                    // macro-steps of bitonic sort double each iteration and determine the size of the
                    // block that should be sorted. We begin by sorting pairs and eventually sort the whole list
                    for (int group_size = merge_only ? warp_size : 2; group_size <= warp_size; group_size = group_size << 1)
                    {
                        // On every micro-step, we merge sorted runs of half the size as before
                        for (int stride = group_size / 2; stride > 0; stride = stride >> 1)
                        {
                            // the thread with which this one will swap is determined by the stride,
                            // similar to a butterfly pattern. We get both the index with which to swap
                            // and a reference to that value in the input array
                            int const paired_lane = lane_id ^ stride;
                            __syncwarp();
                            element_t const paired_val = __shfl_xor_sync(__activemask(), my_val, stride);

                            if (should_swap< ASCENDING >(lane_id, paired_lane, my_val, paired_val, group_size))
                            {
                                my_val = paired_val;
                            }
                        }
                    }
                }

                /**
                 * Perform bitonic sort with one thread block on up to 1024 elements.
                 * Optionally, run just the final merge phase (i.e., last macro-step)
                 * with a custom alternation frequency
                 */
                __global__
                    void bitonic_block(element_t* data, std::size_t num_elements, std::size_t block_level_group_size = 1, bool merge_only = false)
                {
                    assert("this solution assumes the input size is a power of two" && __popc(num_elements) == 1);

                    // Cooperative thread array with 32 teams of 32 threads
                    __shared__ element_t smem[1024];

                    // Organising with x- and y-dimensions simplifies logic
                    // for warp-level stuff, which can just use a simple lane id
                    int const warp_size = blockDim.x;
                    int const block_size = warp_size * blockDim.y;
                    int const lane_id = threadIdx.x;
                    int const warp_id = threadIdx.y;
                    int const th_id = warp_id * blockDim.x + lane_id;
                    int const datum = blockIdx.x * block_size + th_id;

                    // ensure __syncthreads() is safe later by killing all threads
                    // that aren't doing something productive.
                    if (datum >= num_elements) { return; }

                    // work in shared memory instead
                    // copy this 1024x1 tile.
                    element_t my_val = data[datum];
                    smem[th_id] = my_val;
                    __syncthreads();

                    // macro-steps of bitonic sort double each iteration and determine the size of the
                    // block that should be sorted. We begin by sorting pairs and eventually sort the whole list
                    for (int group_size = merge_only ? block_size : warp_size; group_size <= block_size; group_size = group_size << 1)
                    {
                        // On every micro-step, we merge sorted runs of half the size as before
                        for (int stride = group_size / 2; stride >= warp_size; stride = stride >> 1)
                        {
                            // the thread with which this one will swap is determined by the stride,
                            // similar to a butterfly pattern. We get both the index with which to swap
                            // and a reference to that value in the input array
                            int paired_thread = th_id ^ stride;
                            element_t const paired_val = smem[paired_thread];

                            // Normal logic, except that we use the optional alternation pattern.
                            if ((((blockIdx.x & block_level_group_size)) && should_swap< false >(th_id, paired_thread, my_val, paired_val, group_size))
                                || ((!(blockIdx.x & block_level_group_size)) && should_swap< true  >(th_id, paired_thread, my_val, paired_val, group_size)))
                            {
                                // update smem and the register too
                                // saves reading smem later by this thread
                                // but also shares the update with other threads.
                                smem[th_id] = paired_val;
                                my_val = paired_val;
                            }
                            __syncthreads(); // need to ensure all swaps are complete before the next step
                        }

                        // Run last 5 iterations directly in registers
                        // Sort DESC if exactly one of these conditions is true:
                        //    * the optional block-level alternation pattern should be DESC
                        //    * the internal thread-level alternation pattern should be DESC
                        if (__popc(blockIdx.x & block_level_group_size) ^ __popc(th_id & group_size))
                            bitonic_warp< false >(my_val, group_size > warp_size);
                        else
                            bitonic_warp< true  >(my_val, group_size > warp_size);

                        // sync up shared memory again after warp-level solution
                        // so that it can be read by other threads on the next iteration.
                        smem[th_id] = my_val;
                        __syncthreads();
                    }

                    // copy final result back to global memory to share
                    // across thread blocks.
                    data[datum] = my_val;
                }

                /**
                 * Merges values for a bitonic sort using a stride that is larger than the
                 * size of the thread block.
                 */
                __global__
                    void merge_blocks(element_t* data, std::size_t num_elements, std::size_t group_size)
                {
                    // Store transposed data in shared memory
                    __shared__ element_t smem[1024];

                    int const block_size = blockDim.x;
                    int const th_id = threadIdx.x;
                    int const datum = block_size * th_id + blockIdx.x; // note stride for transpose

                    // Simplify indexes to look like normal bitonic sort
                    // by dividing by the fixed block size
                    int const local_group_size = group_size / block_size;

                    // guard to ensure no extra threads launched.
                    if (datum >= num_elements) { return; }

                    // copy transposed data to smem
                    element_t my_val = data[datum];
                    smem[th_id] = my_val;
                    __syncthreads();

                    // Run bitonic sort on the transposed columns
                    for (int stride = local_group_size / 2; stride > 0; stride = stride >> 1)
                    {
                        int  const paired_thread = th_id ^ stride;
                        auto const paired_val = smem[paired_thread];

                        if (should_swap< true  >(th_id, paired_thread, my_val, paired_val, local_group_size))
                        {
                            smem[th_id] = paired_val;
                            my_val = paired_val;
                        }
                        __syncthreads(); // need to ensure all swaps are complete before the next step
                    }

                    // Store result back to gmem, unwinding the transpose
                    data[datum] = my_val;
                }

                /**
                 * Reverses the order of an array of num_elements elements at data
                 */
                __global__
                    void reverse(element_t* data, std::size_t num_elements)
                {
                    // Key idea:
                    // Use two threads within the same block so that we can sync them
                    // after the reads and ensure there are no race conditions.
                    // Launch blocks with 2 threadIdx.y values so that it is easy
                    // to pair up threads based on whether they have the same index or not.

                    // Get the index for this thread and also the index it should swap with
                    int const pair_id = blockIdx.x * blockDim.x + threadIdx.x;
                    int const pair_with = num_elements - 1 - pair_id;

                    if (pair_id < num_elements / 2) // don't go out of bounds
                    {
                        // Two threads should have identical pair_id and pair_with vals
                        // but one will have threadIdx.y = 0 and the other, 1.
                        // Have them read each others values and then write it to
                        // the opposing addresses.
                        auto const my_val = threadIdx.y ? data[pair_id] : data[pair_with];
                        int  const write_to = threadIdx.y ? pair_with : pair_id;
                        __syncthreads();

                        data[write_to] = my_val;
                    }
                }

                /**
                 * This solution uses a team of 1024 blocks, each containing a team of 32 x 32 threads,
                 * in order to sort 2^20 elements in a few milliseconds.
                 */
                void opposing_sort(element_t* data, std::size_t num_elements)
                {
                    assert("this solution works up to 2^20" && num_elements <= 1024 * 1024);
                    assert("this solution expects at lest 2^10" && num_elements >= 128);

                    std::size_t const warp_size = 32;
                    std::size_t const block_size = 1024;
                    std::size_t const num_blocks = (num_elements + block_size - 1) / block_size;

                    // Overall idea:
                    // Consider the array to be a 1024x1024 matrix (conceptually).
                    // 1. We can use a block-level bitonic sort to sort each row,
                    //    though we have to remember to sort them in alternating directions
                    // 2. Then, for steps larger than 1024, we perform the bitonic merge on
                    //    the transpose of the matrix instead of the rows. This is equivalent
                    //    to a stride of some multiple of 1024. For example:
                    //      if on step 4096, then for each column we merge values at row 0/3,
                    //      row 1/4, etc.
                    // 3. After merging across rows and getting back down to 1024, then we
                    //    can use just the last merge step on each row again

                    // Initialise every row to be sorted (asc for even block ids, desc for odd)
                    bitonic_block << < num_blocks, dim3{ warp_size, block_size / warp_size, 1 } >> > (data, num_elements);

                    // Perform merges for steps larger than 1024
                    for (std::size_t group_size = block_size << 1; group_size <= num_elements; group_size = group_size << 1)
                    {
                        // Before the across-row merges all the way down to a step size of 1024
                        // sync the device to ensure any writes to gmem are complete before launching the kernel
                        cudaDeviceSynchronize();
                        merge_blocks << < block_size, num_blocks >> > (data, num_elements, group_size);

                        // Handle the last lg 1024 steps (the ones within a row) by calling
                        // the block-level bitonic sort, but starting on the last macro-step
                        cudaDeviceSynchronize();
                        bitonic_block << < num_blocks, dim3{ warp_size, block_size / warp_size, 1 } >> > (data, num_elements, group_size / block_size, true);
                    }

                    // Finally, call a kernel to reverse the last 3/4ths of the array
                    // Offset the pointer/address to make the kernel simpler.
                    cudaDeviceSynchronize();
                    reverse << < num_blocks, dim3{ block_size >> 1, 2, 1 } >> > (data + 3 * (num_elements >> 2), num_elements >> 2);
                }

            } // namespace hierarchical



            /**
             * Performs all the logic of allocating device vectors and copying host/input
             * vectors to the device. Times the opposing_sort() kernel with wall time,
             * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
             */
            void run_gpu_soln(std::vector< element_t > data, std::size_t switch_at, std::size_t n)
            {
                // Kernel launch configurations. Feel free to change these.
                // This is set to maximise the size of a thread block on a T4, but it hasn't
                // been tuned. It's not known if this is optimal.
                std::size_t const threads_per_block = 1024;
                std::size_t const num_blocks = (n + threads_per_block - 1) / threads_per_block;

                // Allocate arrays on the device/GPU
                element_t* d_data;
                cudaMalloc((void**)&d_data, sizeof(element_t) * n);
                CHECK_ERROR("Allocating input array on device");

                // Copy the input from the host to the device/GPU
                cudaMemcpy(d_data, data.data(), sizeof(element_t) * n, cudaMemcpyHostToDevice);
                CHECK_ERROR("Copying input array to device");

                // Warm the cache on the GPU for a more fair comparison
                warm_the_gpu << < num_blocks, threads_per_block >> > (d_data, switch_at, n);

                // Time the execution of the kernel that you implemented
                auto const kernel_start = std::chrono::high_resolution_clock::now();
                //single_block::opposing_sort<<< num_blocks, threads_per_block, threads_per_block * sizeof( element_t ) >>>( d_data, n );
                hierarchical::opposing_sort(d_data, n);
                cudaDeviceSynchronize();
                auto const kernel_end = std::chrono::high_resolution_clock::now();
                CHECK_ERROR("Executing kernel on device");

                // After the timer ends, copy the result back, free the device vector,
                // and echo out the timings and the results.
                cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
                CHECK_ERROR("Transferring result back to host");
                cudaFree(d_data);
                CHECK_ERROR("Freeing device memory");

                std::cout << "GPU Solution time: "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
                    << " ns" << std::endl;

                for (auto const x : data) std::cout << x << " "; std::cout << std::endl;
            }

        } // namespace gpu
    } // namespace a1
} // namespace csc485b