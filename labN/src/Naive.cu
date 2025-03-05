#include "NaiveGemv.hpp"
#include <iostream>
#include "CpuGemv.hpp"
#include "Validate.hpp"
#include "Config.hpp"
#include <vector>
#include <chrono>


__global__ void gemv_naive(double* matrix, double* vec, long M, long N, double* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) { return; }
    for (int j = 0; j < N; j++) {
        out[i] += matrix[i * N + j] * vec[j];
    }
}


void runNaiveOnce(
    const KernelConfig kernelConfig, 
    const Matrix2D<double>& mat,
    const Array<double>& vec,
    Array<double>& vecOut
) {

    double* cudaMat = NULL;
    double* cudaVec = NULL;
    double* cudaOut = NULL;

    cudaMalloc(&cudaMat, mat.byteSize());
    cudaMalloc(&cudaVec, vec.byteSize());
    cudaMalloc(&cudaOut, vecOut.byteSize());

    cudaMemcpy(cudaMat, mat.dataAs1DArray(), mat.byteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVec, vec.data(), vec.byteSize(), cudaMemcpyHostToDevice);

    gemv_naive<<<kernelConfig.blockCount, kernelConfig.threadCount>>>(cudaMat, cudaVec, mat.rowCount(), mat.colCount(), cudaOut);

    cudaMemcpy(vecOut.data(), cudaOut, vecOut.byteSize(), cudaMemcpyDeviceToHost);

#if PRINTING
        std::cout << "output: " << vecOut << std::endl;
#endif 

    cudaFree(cudaMat);
    cudaFree(cudaVec);
    cudaFree(cudaOut);

}


void runNaive() {

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

        runNaiveOnce(config.kernelConfig, *mat, *vec, *vecOut);

#if CORRECTNESS_CHECK
        validate(*vecOut, *vecOutCPU);
#endif

        delete mat;
        delete vec;
        delete vecOut;
#if CORRECTNESS_CHECK
        delete vecOutCPU;
#endif 

    }

}