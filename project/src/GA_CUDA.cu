#include "GA_CUDA.h"
#include "GA_CPU.h"  // 为占位调用 CPU 版本函数（仅用于部分占位实现）
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
// CUDA 版本适应度计算（单个个体）占位：调用 updatePopulationFitnessCUDA 后，再返回对应个体适应度
//---------------------------------------------------------------------
float computeFitnessCUDA(const Individual &ind, const TSP &tsp) {
    // 这里不单独计算单个个体，而是建议使用批量更新后查找
    // 作为占位，调用 CPU 版本
    return GA::computeFitnessCPU(ind, tsp);
}

//---------------------------------------------------------------------
// CUDA 版本更新种群适应度（父代）：批量计算 tsp.population 中所有个体的 fitness
//---------------------------------------------------------------------
void updatePopulationFitnessCUDA(TSP &tsp) {
    // 获取种群数据（所有岛展平为连续数组）
    int *d_population = tsp.transferPopulationToDevice();          // 长度：popSize * numCities
    float *d_distanceMatrix = tsp.transferDistanceMatrixToDevice();  // 长度：numCities * numCities

    int totalIndividuals = tsp.popSize; // tsp.popSize 为所有岛个体总数
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

    // 根据 TSP::transferPopulationToDevice() 中的展平顺序更新每个个体的 fitness
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

//---------------------------------------------------------------------
// CUDA 版本更新后代适应度：只对 offspring 进行适应度计算
//---------------------------------------------------------------------
    void updateOffspringFitnessCUDA(TSP &tsp, Offspring &offspring) {
    // 计算所有后代个体数量
    int totalOffspring = 0;
    for (int island = 0; island < offspring.size(); island++) {
        totalOffspring += offspring[island].size();
    }

    // 将 offspring 展平到连续数组中，每个个体的染色体长度为 tsp.numCities
    std::vector<int> flatOffspring;
    flatOffspring.reserve(totalOffspring * tsp.numCities);
    for (int island = 0; island < offspring.size(); island++) {
        for (int j = 0; j < offspring[island].size(); j++) {
            const Individual &ind = offspring[island][j];
            // 假定 ind.chromosome 的长度等于 tsp.numCities
            for (int k = 0; k < tsp.numCities; k++) {
                flatOffspring.push_back(ind.chromosome[k]);
            }
        }
    }

    // 将展平后的后代数据复制到设备内存
    int *d_offspring;
    cudaMalloc(&d_offspring, flatOffspring.size() * sizeof(int));
    cudaMemcpy(d_offspring, flatOffspring.data(), flatOffspring.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 获取距离矩阵设备内存
    float *d_distanceMatrix = tsp.transferDistanceMatrixToDevice();

    // 为后代适应度分配设备内存
    float *d_fitness;
    cudaMalloc(&d_fitness, totalOffspring * sizeof(float));

    int threads = 256;
    int blocks = (totalOffspring + threads - 1) / threads;
    computeFitnessKernel<<<blocks, threads>>>(d_offspring, d_distanceMatrix, d_fitness,
                                              tsp.numCities, totalOffspring);
    cudaDeviceSynchronize();

    // 将后代适应度结果复制回 host
    std::vector<float> h_fitness(totalOffspring);
    cudaMemcpy(h_fitness.data(), d_fitness, totalOffspring * sizeof(float), cudaMemcpyDeviceToHost);

    // 更新 offspring 中每个个体的 fitness
    int offset = 0;
    for (int island = 0; island < offspring.size(); island++) {
        for (int j = 0; j < offspring[island].size(); j++) {
            offspring[island][j].fitness = h_fitness[offset];
            offset++;
        }
    }

    cudaFree(d_offspring);
    cudaFree(d_distanceMatrix);
    cudaFree(d_fitness);
}


//---------------------------------------------------------------------
// 以下各模块占位实现，实际需调用 CUDA 内核实现
//---------------------------------------------------------------------
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
