#include <iostream>
#include "tsp.h"
#include <cuda_runtime.h>
#include <vector>
#include "TSP_individual.hpp"

#define THREADS_PER_BLOCK 256


int main() {
    // 参数设置
    int numCities = 10;
    int popSize = 20;
    int mapSize = 100;

    // 在host上实例化TSP对象，初始化数据
    TSP tsp(numCities, popSize, mapSize);

    // 将数据传输到device
    float* d_distanceMatrix = tsp.transferDistanceMatrixToDevice();
    int* d_population = tsp.transferPopulationToDevice();

    // 为适应度分配device内存
    float* d_fitness;
    cudaMalloc(&d_fitness, popSize * sizeof(float));

    int blocks = (popSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fitnessKernel<<<blocks, THREADS_PER_BLOCK>>>(d_population, d_distanceMatrix,
                                                 d_fitness, numCities, popSize);
    cudaDeviceSynchronize();

    // 将结果拷回host并输出
    std::vector<float> h_fitness(popSize);
    cudaMemcpy(h_fitness.data(), d_fitness, popSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "Fitness(distance)" << std::endl;
    for (int i = 0; i < popSize; i++) {
        std::cout << "Individual " << i << ": " << h_fitness[i] << std::endl;
    }

    // 释放device内存
    cudaFree(d_distanceMatrix);
    cudaFree(d_population);
    cudaFree(d_fitness);

    return 0;
}