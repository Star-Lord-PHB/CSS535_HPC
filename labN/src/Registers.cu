#include <iostream>
#include "CpuGemv.hpp"
#include "Validate.hpp"
#include "Config.hpp"
#include <vector>
#include "NaiveGemv.hpp"
#include "Registers.hpp"


__global__ void gemv_register(double* matrix, double* vec, const long M, const long N, const long tileSize, double* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) { return; }
    auto j = 0;
    for (; j + 3 < N; j+=4) {
        out[i] += matrix[i * N + j] * vec[j];
        out[i] += matrix[i * N + j + 1] * vec[j + 1];
        out[i] += matrix[i * N + j + 2] * vec[j + 2];
        out[i] += matrix[i * N + j + 3] * vec[j + 3];
    }
    for (; j < N; j++) {
        out[i] += matrix[i * N + j] * vec[j];
    }
}


void runOnce(
    const KernelConfig kernelConfig, 
    const Matrix2D<double>& mat,
    const Array<double>& vec,
    Array<double>& vecOut
) {

    double* cudaMat = NULL;
    double* cudaVec = NULL;
    double* cudaOut = NULL;

    cudaError_t error;

    cudaMalloc(&cudaMat, mat.byteSize());
    cudaMalloc(&cudaVec, vec.byteSize());
    cudaMalloc(&cudaOut, vecOut.byteSize());

    cudaMemcpy(cudaMat, mat.dataAs1DArray(), mat.byteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVec, vec.data(), vec.byteSize(), cudaMemcpyHostToDevice);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "mem error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }

    const auto tileSize = std::min(mat.colCount(), (long)4096);
    // const auto tileSize = 3;
    gemv_register<<<kernelConfig.blockCount, kernelConfig.threadCount>>>(
        cudaMat, cudaVec, mat.rowCount(), mat.colCount(), tileSize, cudaOut
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "kernel error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaMemcpy(vecOut.data(), cudaOut, vecOut.byteSize(), cudaMemcpyDeviceToHost);

#if PRINTING
        std::cout << "output: " << vecOut << std::endl;
#endif 

    cudaFree(cudaMat);
    cudaFree(cudaVec);
    cudaFree(cudaOut);

}


void runRegisters() {

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
            KernelConfig(dim3(10, 1, 1), dim3(1024, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(10, 1, 1), dim3(1000, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(100, 1, 1), dim3(128, 1, 1)), 
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

        runOnce(config.kernelConfig, *mat, *vec, *vecOut);

#if CORRECTNESS_CHECK
        validate(*vecOut, *vecOutCPU);
#endif
        vecOut->fill(0);

        auto vecOut1 = new Array<double>(config.M);
        runNaiveOnce(config.kernelConfig, *mat, *vec, *vecOut1);

        delete mat;
        delete vec;
        delete vecOut;
#if CORRECTNESS_CHECK
        delete vecOutCPU;
#endif 

    }

}
