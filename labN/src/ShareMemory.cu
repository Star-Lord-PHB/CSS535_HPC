#include <iostream>
#include "CpuGemv.hpp"
#include "Validate.hpp"
#include "Config.hpp"
#include <vector>
#include "NaiveGemv.hpp"
#include "ShareMemory.hpp"


__global__ void gemv_share_memory(double* matrix, double* vec, const long M, const long N, const long tileSize, double* out, double* debugOut) {

    extern __shared__ double row[];

    const auto rowId = blockIdx.x * blockDim.x + threadIdx.x;

    const auto baseElementCountPerThread = tileSize / blockDim.x;
    const auto overflow = tileSize % blockDim.x;

    auto sum = .0;

    for (auto tileId = 0; tileId * tileSize < N; tileId++) {

        // this is to make sure that the threads responsible for copying elements to shared memory but 
        // not responsible for calculation will wait until the other threads to finish calculation before 
        // copying the next tile. 
        __syncthreads();

        const auto copyDestStartOffset = threadIdx.x * baseElementCountPerThread + (threadIdx.x < overflow ? threadIdx.x : overflow);
        const auto copySrcStartOffset = tileId * tileSize + copyDestStartOffset;
        const auto elementLeftForCopy = N - copySrcStartOffset;
        auto elementCountToCopy = baseElementCountPerThread + (threadIdx.x < overflow ? 1 : 0);
        elementCountToCopy = elementCountToCopy > elementLeftForCopy ? elementLeftForCopy : elementCountToCopy;

        for (auto i = 0; i < elementCountToCopy; i++) {
            row[copyDestStartOffset + i] = vec[copySrcStartOffset + i];
        }

        // this is to make sure that the tile in shared memory has all elements loaded 
        __syncthreads();

        // only the first M threads are responsible for actual calculation 
        if (rowId >= M) { continue; }

        const auto rowStartOffset = tileId * tileSize;
        const auto matrixStartOffset = rowId * N + rowStartOffset;
        const auto elementLeftForCalc = N - rowStartOffset;
        const auto elementCountToCalc = tileSize < elementLeftForCalc ? tileSize : elementLeftForCalc;

        for (auto i = 0; i < elementCountToCalc; i++) {
            sum += matrix[matrixStartOffset + i] * row[i];
        }

    }

    if (rowId >= M) { return; }

    out[rowId] = sum;

}



void runOnce(
    const KernelConfig kernelConfig, 
    const Matrix2D<double>& mat,
    const Array<double>& vec,
    Array<double>& vecOut,
    Array<double>& debugOut
) {

    double* cudaMat = NULL;
    double* cudaVec = NULL;
    double* cudaOut = NULL;
    double* cudaDebugOut = NULL;

    cudaError_t error;

    cudaMalloc(&cudaMat, mat.byteSize());
    cudaMalloc(&cudaVec, vec.byteSize());
    cudaMalloc(&cudaOut, vecOut.byteSize());

    cudaMalloc(&cudaDebugOut, debugOut.byteSize());

    cudaMemcpy(cudaMat, mat.dataAs1DArray(), mat.byteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVec, vec.data(), vec.byteSize(), cudaMemcpyHostToDevice);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "mem error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }

    const auto tileSize = std::min(mat.colCount(), (long)4096);
    // const auto tileSize = 3;
    gemv_share_memory<<<kernelConfig.blockCount, kernelConfig.threadCount, tileSize * sizeof(double)>>>(
        cudaMat, cudaVec, mat.rowCount(), mat.colCount(), tileSize, cudaOut, cudaDebugOut
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "kernel error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaMemcpy(vecOut.data(), cudaOut, vecOut.byteSize(), cudaMemcpyDeviceToHost);
    cudaMemcpy(debugOut.data(), cudaDebugOut, debugOut.byteSize(), cudaMemcpyDeviceToHost);

#if PRINTING
        std::cout << "output: " << vecOut << std::endl;
#endif 

    cudaFree(cudaMat);
    cudaFree(cudaVec);
    cudaFree(cudaOut);
    cudaFree(cudaDebugOut);

}



void runShareMemory() {

    const std::vector<Config> configs = {
        Config(
            KernelConfig(dim3(10000, 1, 1), dim3(1, 1, 1)),
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(313, 1, 1), dim3(32, 1, 1)),
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(100, 1, 1), dim3(128, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(32, 1, 1), dim3(320, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(16, 1, 1), dim3(640, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(10, 1, 1), dim3(1024, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(10, 1, 1), dim3(1000, 1, 1)), 
            10000, 10000
        ),
    };

    for (auto& config : configs) {

        auto mat = new Matrix2D<double>(config.M, config.N);
        auto vec = new Array<double>(config.N);
        auto vecOut = new Array<double>(config.M);

        mat->randomInit();
        vec->randomInit();
        vecOut->fill(0);

#if PRINTING
        std::cout << "mat: " << *mat << std::endl;
        std::cout << "vec: " << *vec << std::endl;
#endif 

#if CORRECTNESS_CHECK
        auto vecOutCPU = new Array<double>(config.M);
        cpu_gemv(*mat, *vec, *vecOutCPU);
#if PRINTING
        std::cout << "cpu: " << *vecOutCPU << std::endl;
#endif
#endif

        auto debugOut = new Array<double>(config.N);
        runOnce(config.kernelConfig, *mat, *vec, *vecOut, *debugOut);
        
        delete debugOut;

#if CORRECTNESS_CHECK
        validate(*vecOut, *vecOutCPU);
#endif
        vecOut->fill(0);

        runNaiveOnce(config.kernelConfig, *mat, *vec, *vecOut);

        delete mat;
        delete vec;
        delete vecOut;
#if CORRECTNESS_CHECK
        delete vecOutCPU;
#endif 

    }

}