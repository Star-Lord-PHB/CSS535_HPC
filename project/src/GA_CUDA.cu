#include "GA_CUDA.h"
#include "GA_CPU.h"  // 为占位调用 CPU 版本函数（仅用于缺省实现）
#include "TSP.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>

namespace GA {

//---------------------------------------------------------------------
// CUDA 内核：批量计算种群适应度
//---------------------------------------------------------------------
__global__ void computeFitnessKernel(const int *d_population,
                                     const float *d_distanceMatrix,
                                     float *d_fitness,
                                     int numCities,
                                     int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popSize) {
        float totalDistance = 0.0f;
        for (int i = 0; i < numCities - 1; i++) {
            int city1 = d_population[idx * numCities + i];
            int city2 = d_population[idx * numCities + i + 1];
            totalDistance += d_distanceMatrix[city1 * numCities + city2];
        }
        int firstCity = d_population[idx * numCities];
        int lastCity  = d_population[idx * numCities + numCities - 1];
        totalDistance += d_distanceMatrix[lastCity * numCities + firstCity];
        d_fitness[idx] = (totalDistance <= 0.0f) ? 0.0f : 1.0f / totalDistance;
    }
}

//---------------------------------------------------------------------
// CUDA 版本适应度计算（单个个体）占位：调用 updateFitnessCUDA 后，再返回对应个体适应度
//---------------------------------------------------------------------
float computeFitnessCUDA(const Individual &ind, const TSP &tsp) {
    // 这里不单独计算单个个体，而是建议使用 updateFitnessCUDA 更新整个种群后查找
    // 作为占位，调用 CPU 版本
    return GA::computeFitnessCPU(ind, tsp);
}

//---------------------------------------------------------------------
// CUDA 版本更新整个种群适应度
//---------------------------------------------------------------------
void updateFitnessCUDA(TSP &tsp) {
    // 将种群平铺为连续数组，利用 TSP 类提供的接口
    int *d_population = tsp.transferPopulationToDevice();      // 长度：popSize * numCities
    float *d_distanceMatrix = tsp.transferDistanceMatrixToDevice(); // 长度：numCities * numCities

    int totalIndividuals = tsp.popSize; // 注意：popSize 为所有岛个体总数
    float *d_fitness;
    cudaMalloc(&d_fitness, totalIndividuals * sizeof(float));

    int threads = 256;
    int blocks = (totalIndividuals + threads - 1) / threads;
    computeFitnessKernel<<<blocks, threads>>>(d_population, d_distanceMatrix, d_fitness,
                                              tsp.numCities, totalIndividuals);
    cudaDeviceSynchronize();

    // 拷贝计算结果回 host
    std::vector<float> h_fitness(totalIndividuals);
    cudaMemcpy(h_fitness.data(), d_fitness, totalIndividuals * sizeof(float), cudaMemcpyDeviceToHost);

    // 根据 TSP::transferPopulationToDevice() 中的平铺顺序，更新每个个体的 fitness
    int offset = 0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (int i = 0; i < tsp.population[island].size(); i++) {
            tsp.population[island][i].fitness = h_fitness[offset];
            offset++;
        }
    }

    cudaFree(d_population);
    cudaFree(d_distanceMatrix);
    cudaFree(d_fitness);
}

// 以下各模块占位实现，实际需调用 CUDA 内核实现

ParentPairs selectionCUDA(TSP &tsp) {
    std::cout << "[CUDA] selection (placeholder)\n";
    return selectionCPU(tsp);
}

Offspring crossoverCUDA(const TSP &tsp, const ParentPairs &parentPairs) {
    std::cout << "[CUDA] crossover (placeholder)\n";
    return crossoverCPU(tsp, parentPairs);
}

void mutationCUDA(const TSP &tsp, Offspring &offspring) {
    std::cout << "[CUDA] mutation (placeholder)\n";
    mutationCPU(tsp, offspring);
}

void replacementCUDA(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring) {
    std::cout << "[CUDA] replacement (placeholder)\n";
    replacementCPU(tsp, parentPairs, offspring);
}

void migrationCUDA(TSP &tsp) {
    std::cout << "[CUDA] migration (placeholder)\n";
    migrationCPU(tsp);
}

} // namespace GA
