#include "tma_bandwidth.cuh"

#ifdef TEST_THREAD_MEMORY_TILE_TMA_BW

#include <algorithm>
#include <iomanip>
#include <limits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

#ifndef TMA_BW_TILE_H
#define TMA_BW_TILE_H 4
#endif

#ifndef TMA_BW_TILE_W
#define TMA_BW_TILE_W 4
#endif

#ifndef TMA_BW_WARPS_PER_BLOCK
#define TMA_BW_WARPS_PER_BLOCK 4
#endif

#ifndef TMA_BW_OPS_PER_ITER
#define TMA_BW_OPS_PER_ITER 4
#endif

#ifndef TMA_BW_ITERS
#define TMA_BW_ITERS 2000
#endif

#ifndef TMA_BW_WARMUP
#define TMA_BW_WARMUP 2
#endif

#ifndef TMA_BW_REPEATS
#define TMA_BW_REPEATS 5
#endif

#ifndef TMA_BW_SATURATION
#define TMA_BW_SATURATION 0.90f
#endif

template <typename T>
__device__ inline float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float to_float<kittens::bf16>(kittens::bf16 v) {
    return __bfloat162float(v);
}

template <>
__device__ inline float to_float<kittens::half>(kittens::half v) {
    return __half2float(v);
}

template <typename T, int H, int W>
using tile_t = kittens::st<T, 16 * H, 16 * W>;

template <typename T, int H, int W>
using gl_t = kittens::gl<T, -1, -1, H * 16, W * 16, tile_t<T, H, W>>;

template <typename T, int H, int W, int WARPS, int OPS>
__global__ void tma_load_bw_kernel(const __grid_constant__ gl_t<T, H, W> input,
                                   float *sink,
                                   int num_tiles,
                                   int iters) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int *)&__shm[0]);
    using tile_type = tile_t<T, H, W>;
    tile_type (&tiles)[WARPS][OPS] = al.allocate<tile_type, WARPS, OPS>();

    __shared__ kittens::semaphore sem[WARPS];

    const int warp_id = kittens::warpid();
    const int lane = kittens::laneid();
    if (warp_id < WARPS && lane == 0) {
        kittens::warp::init_semaphore(sem[warp_id], 0, 1);
    }
    __syncthreads();

    if (warp_id >= WARPS) {
        return;
    }

    constexpr uint32_t bytes_per_iter =
        static_cast<uint32_t>(sizeof(T) * tile_type::num_elements * OPS);

    float acc = 0.0f;
    const int warp_stride = WARPS * OPS;
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * warp_stride;

    for (int iter = 0; iter < iters; ++iter) {
        int64_t base = static_cast<int64_t>(blockIdx.x) * warp_stride +
                       warp_id * OPS +
                       static_cast<int64_t>(iter) * grid_stride;

        if (lane == 0) {
            kittens::tma::expect_bytes(sem[warp_id], bytes_per_iter);
        }

        #pragma unroll
        for (int op = 0; op < OPS; ++op) {
            int tile = static_cast<int>((base + op) % num_tiles);
            if (lane == 0) {
                kittens::tma::load_async(tiles[warp_id][op], input, {tile, 0, 0, 0}, sem[warp_id]);
            }
        }

        kittens::wait(sem[warp_id], iter & 1);

        if (lane == 0) {
            acc += to_float(tiles[warp_id][0][0]);
        }
    }

    if (lane == 0) {
        sink[blockIdx.x * WARPS + warp_id] = acc;
    }
}

template <typename T, int H, int W, int WARPS, int OPS>
__global__ void tma_store_bw_kernel(const __grid_constant__ gl_t<T, H, W> output,
                                    int num_tiles,
                                    int iters) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int *)&__shm[0]);
    using tile_type = tile_t<T, H, W>;
    tile_type (&tiles)[WARPS][OPS] = al.allocate<tile_type, WARPS, OPS>();

    const int warp_id = kittens::warpid();
    const int lane = kittens::laneid();
    if (warp_id < WARPS) {
        kittens::rt<T, 16 * H, 16 * W> reg_tile;
        kittens::warp::one(reg_tile);
        #pragma unroll
        for (int op = 0; op < OPS; ++op) {
            kittens::warp::store(tiles[warp_id][op], reg_tile);
        }
    }
    __syncthreads();

    if (warp_id >= WARPS) {
        return;
    }

    const int warp_stride = WARPS * OPS;
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * warp_stride;

    for (int iter = 0; iter < iters; ++iter) {
        int64_t base = static_cast<int64_t>(blockIdx.x) * warp_stride +
                       warp_id * OPS +
                       static_cast<int64_t>(iter) * grid_stride;

        #pragma unroll
        for (int op = 0; op < OPS; ++op) {
            int tile = static_cast<int>((base + op) % num_tiles);
            if (lane == 0) {
                kittens::tma::store_async(output, tiles[warp_id][op], {tile, 0, 0, 0});
            }
        }
        if (lane == 0) {
            kittens::tma::store_async_read_wait<OPS - 1>();
        }
    }

    if (lane == 0) {
        kittens::tma::store_async_read_wait();
    }
}

template <typename Kernel, typename... Args>
float measure_kernel_ms(Kernel kernel,
                        dim3 grid,
                        dim3 block,
                        size_t shared_bytes,
                        int warmup,
                        int repeats,
                        Args... args) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block, shared_bytes>>>(args...);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float best_ms = std::numeric_limits<float>::max();
    for (int i = 0; i < repeats; ++i) {
        cudaEventRecord(start);
        kernel<<<grid, block, shared_bytes>>>(args...);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        best_ms = std::min(best_ms, ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return best_ms;
}

struct sweep_result {
    int sm_count;
    int blocks_for_saturation;
    double max_gbps;
};

template <typename LaunchFn>
sweep_result run_sweep(const char *label,
                       int sm_count,
                       int warps_per_block,
                       int ops_per_iter,
                       size_t tile_bytes,
                       int iters,
                       float saturation_ratio,
                       const LaunchFn &launch) {
    std::cout << "\n" << label << "\n";
    std::cout << "blocks, sms, gbps\n";

    double max_gbps = 0.0;
    int max_blocks = 0;
    std::vector<double> bw(sm_count + 1, 0.0);

    for (int blocks = 1; blocks <= sm_count; ++blocks) {
        float ms = launch(blocks);
        double bytes = static_cast<double>(blocks) * warps_per_block * ops_per_iter *
                       static_cast<double>(tile_bytes) * iters;
        double gbps = bytes / (ms * 1e-3) / 1e9;
        bw[blocks] = gbps;
        if (gbps > max_gbps) {
            max_gbps = gbps;
            max_blocks = blocks;
        }
        std::cout << std::setw(6) << blocks << ", "
                  << std::setw(4) << blocks << ", "
                  << std::fixed << std::setprecision(2) << std::setw(8) << gbps << "\n";
    }

    int sat_blocks = max_blocks;
    const double target = max_gbps * saturation_ratio;
    for (int blocks = 1; blocks <= sm_count; ++blocks) {
        if (bw[blocks] >= target) {
            sat_blocks = blocks;
            break;
        }
    }

    std::cout << "peak_gbps=" << std::fixed << std::setprecision(2) << max_gbps
              << " at blocks=" << max_blocks << "\n";
    std::cout << ">= " << std::fixed << std::setprecision(2) << (saturation_ratio * 100.0)
              << "% of peak at blocks=" << sat_blocks << "\n";

    return {sm_count, sat_blocks, max_gbps};
}

} // namespace

void thread::memory::tile::tma_bandwidth::tests(test_data &results) {
    std::cout << " ----- Starting ops/thread/memory/tile/tma_bandwidth test -----\n" << std::endl;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    constexpr int kTileH = TMA_BW_TILE_H;
    constexpr int kTileW = TMA_BW_TILE_W;
    constexpr int kWarps = TMA_BW_WARPS_PER_BLOCK;
    constexpr int kOps = TMA_BW_OPS_PER_ITER;
    constexpr int kIters = TMA_BW_ITERS;
    constexpr int kWarmup = TMA_BW_WARMUP;
    constexpr int kRepeats = TMA_BW_REPEATS;
    constexpr float kSaturation = TMA_BW_SATURATION;

    using dtype = kittens::bf16;
    using tile_type = tile_t<dtype, kTileH, kTileW>;
    using gl_type = gl_t<dtype, kTileH, kTileW>;

    const int sm_count = prop.multiProcessorCount;
    const size_t tile_bytes = sizeof(dtype) * tile_type::num_elements;

    size_t target_bytes = std::max<size_t>(prop.l2CacheSize * 2, 64ull * 1024 * 1024);
    target_bytes = std::min<size_t>(target_bytes, 512ull * 1024 * 1024);
    const int num_tiles = static_cast<int>((target_bytes + tile_bytes - 1) / tile_bytes);
    const size_t num_elems = static_cast<size_t>(num_tiles) * tile_type::num_elements;

    dtype *d_input = nullptr;
    dtype *d_output = nullptr;
    float *d_sink = nullptr;

    cudaMalloc(&d_input, num_elems * sizeof(dtype));
    cudaMalloc(&d_output, num_elems * sizeof(dtype));
    cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * kWarps * sizeof(float));
    cudaMemset(d_input, 0, num_elems * sizeof(dtype));
    cudaMemset(d_output, 0, num_elems * sizeof(dtype));
    cudaMemset(d_sink, 0, static_cast<size_t>(sm_count) * kWarps * sizeof(float));

    gl_type input(d_input, num_tiles, 1, nullptr, nullptr);
    gl_type output(d_output, num_tiles, 1, nullptr, nullptr);

    const int threads = kWarps * kittens::WARP_THREADS;
    const size_t shared_bytes = kittens::MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(
        tma_load_bw_kernel<dtype, kTileH, kTileW, kWarps, kOps>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);
    cudaFuncSetAttribute(
        tma_store_bw_kernel<dtype, kTileH, kTileW, kWarps, kOps>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);

    double theoretical_bw = 0.0;
    if (prop.memoryClockRate > 0 && prop.memoryBusWidth > 0) {
        const double mem_clock_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
        const double bus_bytes = static_cast<double>(prop.memoryBusWidth) / 8.0;
        theoretical_bw = 2.0 * mem_clock_hz * bus_bytes / 1.0e9;
    }

    std::cout << "device=" << prop.name << " sm_count=" << sm_count << "\n";
    std::cout << "tile=" << (kTileH * 16) << "x" << (kTileW * 16)
              << " warps_per_block=" << kWarps << " ops_per_iter=" << kOps
              << " iters=" << kIters << "\n";
    std::cout << "footprint_mb=" << static_cast<double>(num_elems * sizeof(dtype)) / (1024.0 * 1024.0)
              << " l2_mb=" << static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0) << "\n";
    if (theoretical_bw > 0.0) {
        std::cout << "theoretical_hbm_gbps=" << std::fixed << std::setprecision(2)
                  << theoretical_bw << "\n";
    }

    auto load_launch = [&](int blocks) {
        return measure_kernel_ms(
            tma_load_bw_kernel<dtype, kTileH, kTileW, kWarps, kOps>,
            dim3(blocks),
            dim3(threads),
            shared_bytes,
            kWarmup,
            kRepeats,
            input,
            d_sink,
            num_tiles,
            kIters);
    };

    auto store_launch = [&](int blocks) {
        return measure_kernel_ms(
            tma_store_bw_kernel<dtype, kTileH, kTileW, kWarps, kOps>,
            dim3(blocks),
            dim3(threads),
            shared_bytes,
            kWarmup,
            kRepeats,
            output,
            num_tiles,
            kIters);
    };

    run_sweep("tma_ld (gmem->smem) bandwidth sweep", sm_count, kWarps, kOps, tile_bytes, kIters, kSaturation, load_launch);
    run_sweep("tma_st (smem->gmem) bandwidth sweep", sm_count, kWarps, kOps, tile_bytes, kIters, kSaturation, store_launch);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sink);

    test_info info;
    info.label = "tma_bandwidth_sweep";
    info.result = test_result::PASSED;
    results.push_back(info);

    std::cout << std::endl;
}

#endif
