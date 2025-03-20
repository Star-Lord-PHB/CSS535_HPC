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
            int lastCity = d_population[idx * numCities + numCities - 1];
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
        int *d_population = tsp.transferPopulationToDevice(); // 长度：popSize * numCities
        float *d_distanceMatrix = tsp.transferDistanceMatrixToDevice(); // 长度：numCities * numCities

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

    // 假设最大城市数不会超过 256
#define MAX_CITIES 256

    //---------------------------------------------------------------------
    // CUDA 内核：对每个父代配对执行顺序交叉（OX）
    //---------------------------------------------------------------------
    __global__ void crossoverKernel(const int *d_parentA, const int *d_parentB,
                                    int *d_child1, int *d_child2,
                                    int numPairs, int numCities,
                                    float crossoverProbability, unsigned long seed) {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx < numPairs) {
            // 初始化每个线程自己的 curand 状态
            curandState state;
            curand_init(seed, pairIdx, 0, &state);

            // 父代指针：每个父代染色体长度为 numCities
            const int *pA = d_parentA + pairIdx * numCities;
            const int *pB = d_parentB + pairIdx * numCities;
            int *c1 = d_child1 + pairIdx * numCities;
            int *c2 = d_child2 + pairIdx * numCities;

            // 决定是否进行交叉
            float r = curand_uniform(&state);
            if (r >= crossoverProbability) {
                // 不交叉，直接复制父代染色体
                for (int i = 0; i < numCities; i++) {
                    c1[i] = pA[i];
                    c2[i] = pB[i];
                }
            } else {
                // 交叉操作：顺序交叉 (OX)
                int point1 = curand(&state) % numCities;
                int point2 = curand(&state) % numCities;
                if (point1 > point2) {
                    int tmp = point1;
                    point1 = point2;
                    point2 = tmp;
                }
                // 使用局部数组存储子代染色体，先全部置为 -1 表示未填充
                int child1[MAX_CITIES];
                int child2[MAX_CITIES];
                for (int i = 0; i < numCities; i++) {
                    child1[i] = -1;
                    child2[i] = -1;
                }
                // 复制交叉区间：子代1复制父代A，子代2复制父代B
                for (int i = point1; i <= point2; i++) {
                    child1[i] = pA[i];
                    child2[i] = pB[i];
                }
                // 填充子代1：从父代B中按顺序填充未复制的基因
                int currentIndex = (point2 + 1) % numCities;
                for (int i = 0; i < numCities; i++) {
                    int candidateIndex = (point2 + 1 + i) % numCities;
                    int candidate = pB[candidateIndex];
                    // 检查 candidate 是否已存在于子代1交叉区间内
                    bool found = false;
                    for (int j = point1; j <= point2; j++) {
                        if (child1[j] == candidate) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        child1[currentIndex] = candidate;
                        currentIndex = (currentIndex + 1) % numCities;
                    }
                }
                // 填充子代2：从父代A中按顺序填充未复制的基因
                currentIndex = (point2 + 1) % numCities;
                for (int i = 0; i < numCities; i++) {
                    int candidateIndex = (point2 + 1 + i) % numCities;
                    int candidate = pA[candidateIndex];
                    bool found = false;
                    for (int j = point1; j <= point2; j++) {
                        if (child2[j] == candidate) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        child2[currentIndex] = candidate;
                        currentIndex = (currentIndex + 1) % numCities;
                    }
                }
                // 将生成的子代写入输出
                for (int i = 0; i < numCities; i++) {
                    c1[i] = child1[i];
                    c2[i] = child2[i];
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // CUDA 版本交叉：对所有父代配对执行顺序交叉，采用个体级并行
    //---------------------------------------------------------------------
    Offspring crossoverCUDA(const TSP &tsp, const ParentPairs &parentPairs) {
        // 1. 将父代配对展平到两个连续数组中
        int totalPairs = 0;
        std::vector<int> h_parentA;
        std::vector<int> h_parentB;
        std::vector<int> pairsPerIsland; // 保存每个岛的配对数量
        for (int island = 0; island < parentPairs.size(); island++) {
            int numPairs = parentPairs[island].size();
            pairsPerIsland.push_back(numPairs);
            totalPairs += numPairs;
            for (int i = 0; i < numPairs; i++) {
                const Individual &pa = parentPairs[island][i].first;
                const Individual &pb = parentPairs[island][i].second;
                for (int j = 0; j < tsp.numCities; j++) {
                    h_parentA.push_back(pa.chromosome[j]);
                    h_parentB.push_back(pb.chromosome[j]);
                }
            }
        }
        int numGenes = tsp.numCities;
        int arraySize = totalPairs * numGenes * sizeof(int);
        // 2. 分配设备内存并复制父代数据
        int *d_parentA, *d_parentB, *d_child1, *d_child2;
        cudaMalloc(&d_parentA, arraySize);
        cudaMalloc(&d_parentB, arraySize);
        cudaMalloc(&d_child1, arraySize);
        cudaMalloc(&d_child2, arraySize);
        cudaMemcpy(d_parentA, h_parentA.data(), arraySize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_parentB, h_parentB.data(), arraySize, cudaMemcpyHostToDevice);

        // 3. 启动 CUDA 内核：每个线程处理一对父代
        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        unsigned long seed = 1234; // 可根据需要设置种子
        crossoverKernel<<<blocks, threads>>>(d_parentA, d_parentB, d_child1, d_child2,
                                             totalPairs, numGenes, tsp.crossoverProbability, seed);
        cudaDeviceSynchronize();

        // 4. 拷贝生成的子代数据回 host
        std::vector<int> h_child1(totalPairs * numGenes);
        std::vector<int> h_child2(totalPairs * numGenes);
        cudaMemcpy(h_child1.data(), d_child1, arraySize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_child2.data(), d_child2, arraySize, cudaMemcpyDeviceToHost);

        // 释放设备内存
        cudaFree(d_parentA);
        cudaFree(d_parentB);
        cudaFree(d_child1);
        cudaFree(d_child2);

        // 5. 根据原有岛的划分重构 offspring 结构
        Offspring offspring;
        offspring.resize(tsp.numIslands);
        int pairIndex = 0;
        for (int island = 0; island < parentPairs.size(); island++) {
            int numPairs = pairsPerIsland[island];
            for (int i = 0; i < numPairs; i++) {
                Individual child1, child2;
                child1.chromosome.resize(numGenes);
                child2.chromosome.resize(numGenes);
                for (int j = 0; j < numGenes; j++) {
                    child1.chromosome[j] = h_child1[pairIndex * numGenes + j];
                    child2.chromosome[j] = h_child2[pairIndex * numGenes + j];
                }
                offspring[island].push_back(child1);
                offspring[island].push_back(child2);
                pairIndex++;
            }
        }
        return offspring;
    }

    // CUDA 内核：每个线程处理一个个体的染色体变异
    __global__ void mutationKernel(int *d_offspring, int totalIndividuals, int numCities,
                                   float mutationProbability, unsigned long seed) {
        int ind = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程对应的个体编号
        if (ind < totalIndividuals) {
            // 初始化每个线程自己的随机数状态
            curandState state;
            curand_init(seed, ind, 0, &state);

            // 计算该个体在展平数组中的起始位置
            int offset = ind * numCities;
            // 遍历该个体的所有基因
            for (int i = 0; i < numCities; i++) {
                float r = curand_uniform(&state);
                if (r < mutationProbability) {
                    // 随机选择一个基因位置进行交换
                    int j = curand(&state) % numCities;
                    // 交换 d_offspring[offset+i] 和 d_offspring[offset+j]
                    int tmp = d_offspring[offset + i];
                    d_offspring[offset + i] = d_offspring[offset + j];
                    d_offspring[offset + j] = tmp;
                }
            }
        }
    }

    void mutationCUDA(const TSP &tsp, Offspring &offspring) {
        // 计算总后代个体数量
        int totalOffspring = 0;
        for (int island = 0; island < offspring.size(); island++) {
            totalOffspring += offspring[island].size();
        }

        // 将 offspring 展平到一个连续数组中，每个个体占 tsp.numCities 个整数
        std::vector<int> flatOffspring;
        flatOffspring.reserve(totalOffspring * tsp.numCities);
        for (int island = 0; island < offspring.size(); island++) {
            for (int j = 0; j < offspring[island].size(); j++) {
                const Individual &ind = offspring[island][j];
                // 假定 ind.chromosome 的大小为 tsp.numCities
                for (int k = 0; k < tsp.numCities; k++) {
                    flatOffspring.push_back(ind.chromosome[k]);
                }
            }
        }

        int arraySize = totalOffspring * tsp.numCities * sizeof(int);
        int *d_offspring;
        cudaMalloc(&d_offspring, arraySize);
        cudaMemcpy(d_offspring, flatOffspring.data(), arraySize, cudaMemcpyHostToDevice);

        // 启动 CUDA 内核，每个线程处理一个个体
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        unsigned long seed = 1234; // 设定随机种子，可根据需要调整
        mutationKernel<<<blocks, threads>>>(d_offspring, totalOffspring, tsp.numCities,
                                            tsp.mutationProbability, seed);
        cudaDeviceSynchronize();

        // 将变异后的数据拷贝回 host
        cudaMemcpy(flatOffspring.data(), d_offspring, arraySize, cudaMemcpyDeviceToHost);

        // 根据展平数组重新构造 offspring 结构
        int offset = 0;
        for (int island = 0; island < offspring.size(); island++) {
            for (int j = 0; j < offspring[island].size(); j++) {
                for (int k = 0; k < tsp.numCities; k++) {
                    offspring[island][j].chromosome[k] = flatOffspring[offset++];
                }
            }
        }
        cudaFree(d_offspring);
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
