// GA_CUDA.cpp
#include "GA_CUDA.h"
#include "GA_CPU.h"  // 部分占位调用 CPU 函数（仅用于 selectionCUDA 的参考）
#include "TSP.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>
#include <iostream>

namespace GA {

    // -----------------------------------------------------------
    // 核函数：计算个体适应度
    // -----------------------------------------------------------
    __global__ void computeFitnessKernel(const int *d_population,
                                           const float *d_distanceMatrix,
                                           float *d_fitness,
                                           int numCities,
                                           int popCount)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < popCount) {
            float totalDist = 0.0f;
            int base = idx * numCities;
            for (int i = 0; i < numCities - 1; i++) {
                int c1 = d_population[base + i];
                int c2 = d_population[base + i + 1];
                totalDist += d_distanceMatrix[c1 * numCities + c2];
            }
            // 回到起点
            int lastCity = d_population[base + numCities - 1];
            int firstCity = d_population[base];
            totalDist += d_distanceMatrix[lastCity * numCities + firstCity];
            d_fitness[idx] = (totalDist <= 0.0f) ? 0.0f : (1.0f / totalDist);
        }
    }

    // -----------------------------------------------------------
    // 核函数：顺序交叉 (OX) 每个线程处理一对父母
    // -----------------------------------------------------------
    __global__ void crossoverKernel(const int *d_parentA, const int *d_parentB,
                                    int *d_child1, int *d_child2,
                                    int numPairs, int numCities,
                                    float crossoverProb, unsigned long seed)
    {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx >= numPairs) return;

        // 每个线程独立初始化 curand 状态
        curandState state;
        curand_init(seed, pairIdx, 0, &state);

        int base = pairIdx * numCities;
        int *child1 = d_child1 + base;
        int *child2 = d_child2 + base;

        float r = curand_uniform(&state);
        if (r >= crossoverProb) {
            // 不交叉，直接复制父代
            for (int i = 0; i < numCities; i++) {
                child1[i] = d_parentA[base + i];
                child2[i] = d_parentB[base + i];
            }
            return;
        }

        // 初始化子代为-1
        for (int i = 0; i < numCities; i++) {
            child1[i] = -1;
            child2[i] = -1;
        }
        // 随机选择两个交叉点
        int p1 = curand(&state) % numCities;
        int p2 = curand(&state) % numCities;
        if (p1 > p2) {
            int tmp = p1; p1 = p2; p2 = tmp;
        }
        // 复制交叉区间
        for (int i = p1; i <= p2; i++) {
            child1[i] = d_parentA[base + i];
            child2[i] = d_parentB[base + i];
        }
        // 填充 child1：从父B中按顺序填充
        int idx = (p2 + 1) % numCities;
        for (int i = 0; i < numCities; i++) {
            int pos = (p2 + 1 + i) % numCities;
            int gene = d_parentB[base + pos];
            bool found = false;
            for (int j = p1; j <= p2; j++) {
                if (child1[j] == gene) { found = true; break; }
            }
            if (!found) {
                child1[idx] = gene;
                idx = (idx + 1) % numCities;
            }
        }
        // 填充 child2：从父A中按顺序填充
        idx = (p2 + 1) % numCities;
        for (int i = 0; i < numCities; i++) {
            int pos = (p2 + 1 + i) % numCities;
            int gene = d_parentA[base + pos];
            bool found = false;
            for (int j = p1; j <= p2; j++) {
                if (child2[j] == gene) { found = true; break; }
            }
            if (!found) {
                child2[idx] = gene;
                idx = (idx + 1) % numCities;
            }
        }
    }

    // -----------------------------------------------------------
    // 核函数：变异，每个线程处理一个个体
    // -----------------------------------------------------------
    __global__ void mutationKernel(int *d_offspring, int totalIndividuals, int numCities,
                                   float mutationProb, unsigned long seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= totalIndividuals) return;

        curandState state;
        curand_init(seed, idx, 0, &state);

        int start = idx * numCities;
        for (int i = 0; i < numCities; i++) {
            float r = curand_uniform(&state);
            if (r < mutationProb) {
                int j = curand(&state) % numCities;
                int tmp = d_offspring[start + i];
                d_offspring[start + i] = d_offspring[start + j];
                d_offspring[start + j] = tmp;
            }
        }
    }

    // -----------------------------------------------------------
    // 核函数：Replacement
    // 每个线程处理一对父母及对应的两个子代，从 {pa, pb, child1, child2} 中选出最优两个
    // -----------------------------------------------------------
    __global__ void replacementKernel(
        int *d_population, float *d_populationFitness,
        const int *d_parentA, const int *d_parentB,
        const float *d_parentFitness,
        const int *d_child1, const int *d_child2,
        const float *d_childFitness,
        int numPairs, int numCities)
    {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx >= numPairs) return;

        int base = pairIdx * numCities;
        // 以下四个数组均长度为 numCities，每个代表一个个体
        // 依次为：pa, pb, child1, child2
        // 同时，各个个体的适应度分别存储在对应的输入数组中（每对两个父代对应两个子代）
        float fits[4] = {
            d_parentFitness[2 * pairIdx],
            d_parentFitness[2 * pairIdx + 1],
            d_childFitness[2 * pairIdx],
            d_childFitness[2 * pairIdx + 1]
        };
        const int* chrom[4] = {
            d_parentA + base,
            d_parentB + base,
            d_child1 + base,
            d_child2 + base
        };

        // 找出两个最大适应度的索引
        int bestIdx = 0, secondIdx = 1;
        if (fits[secondIdx] > fits[bestIdx]) {
            int tmp = bestIdx; bestIdx = secondIdx; secondIdx = tmp;
        }
        for (int i = 2; i < 4; i++) {
            if (fits[i] > fits[bestIdx]) {
                secondIdx = bestIdx;
                bestIdx = i;
            } else if (fits[i] > fits[secondIdx]) {
                secondIdx = i;
            }
        }
        // 用最佳的两个染色体替换种群中对应父代的位置
        int popOffsetA = pairIdx * 2 * numCities;      // 父代 pa 在 population 中的起始位置
        int popOffsetB = popOffsetA + numCities;         // 父代 pb 在 population 中的起始位置
        for (int i = 0; i < numCities; i++) {
            d_population[popOffsetA + i] = chrom[bestIdx][i];
            d_population[popOffsetB + i] = chrom[secondIdx][i];
        }
        // 更新适应度
        d_populationFitness[2 * pairIdx] = fits[bestIdx];
        d_populationFitness[2 * pairIdx + 1] = fits[secondIdx];
    }

    // -----------------------------------------------------------
    // selectionCUDA
    // 使用 CPU 实现并更新 TSP 内展平数据（不另外展平）
    // -----------------------------------------------------------
    ParentPairs selectionCUDA(TSP &tsp) {
        // 采用 CPU 的选择实现，然后更新 TSP.parentAFlat、parentBFlat、parentFitnessFlat
        ParentPairs pp = selectionCPU(tsp);
        tsp.parentPairs = pp;
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += pp[i].size();
        }
        tsp.parentAFlat.resize(totalPairs * tsp.numCities);
        tsp.parentBFlat.resize(totalPairs * tsp.numCities);
        tsp.parentFitnessFlat.resize(2 * totalPairs);
        int pairIndex = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < pp[island].size(); i++) {
                const Individual &pa = pp[island][i].first;
                const Individual &pb = pp[island][i].second;
                for (int j = 0; j < tsp.numCities; j++) {
                    tsp.parentAFlat[pairIndex * tsp.numCities + j] = pa.chromosome[j];
                    tsp.parentBFlat[pairIndex * tsp.numCities + j] = pb.chromosome[j];
                }
                tsp.parentFitnessFlat[2 * pairIndex]     = pa.fitness;
                tsp.parentFitnessFlat[2 * pairIndex + 1] = pb.fitness;
                pairIndex++;
            }
        }
        return pp;
    }

    // -----------------------------------------------------------
    // crossoverCUDA
    // 使用 TSP.parentAFlat 和 parentBFlat，不再重新展平
    // -----------------------------------------------------------
    Offspring crossoverCUDA(TSP &tsp, const ParentPairs &parentPairs) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += parentPairs[i].size();
        }
        // parentAFlat、parentBFlat 已在 selectionCUDA 中更新到 TSP 内
        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        unsigned long seed = time(nullptr);
        // 启动 crossover kernel
        crossoverKernel<<<blocks, threads>>>(tsp.d_parentA, tsp.d_parentB, tsp.d_child1, tsp.d_child2,
                                               totalPairs, tsp.numCities, tsp.crossoverProbability, seed);
        cudaDeviceSynchronize();
        // 直接将 kernel 结果存入 TSP.offspringFlat（大小为 totalPairs * 2 * numCities）
        tsp.offspringFlat.resize(totalPairs * 2 * tsp.numCities);
        cudaMemcpy(tsp.offspringFlat.data(), tsp.d_child1, totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tsp.offspringFlat.data() + totalPairs * tsp.numCities, tsp.d_child2,
                   totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
        // 重建 Offspring 结构（不再额外展平）
        Offspring offspring;
        offspring.resize(tsp.numIslands);
        int pairIndex = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            int numPairs = parentPairs[island].size();
            for (int i = 0; i < numPairs; i++) {
                Individual child1, child2;
                child1.chromosome.resize(tsp.numCities);
                child2.chromosome.resize(tsp.numCities);
                for (int j = 0; j < tsp.numCities; j++) {
                    child1.chromosome[j] = tsp.offspringFlat[pairIndex * tsp.numCities + j];
                    child2.chromosome[j] = tsp.offspringFlat[(totalPairs + pairIndex) * tsp.numCities + j];
                }
                child1.fitness = 0.0f;
                child2.fitness = 0.0f;
                child1.islandID = island;
                child2.islandID = island;
                offspring[island].push_back(child1);
                offspring[island].push_back(child2);
                pairIndex++;
            }
        }
        return offspring;
    }

    // -----------------------------------------------------------
    // mutationCUDA
    // 使用 TSP.offspringFlat 中已有的展平数据，不在每个函数里重新展平
    // -----------------------------------------------------------
    void mutationCUDA(TSP &tsp, Offspring &offspring) {
        // 假设 TSP.offspringFlat 已由 crossoverCUDA 更新
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int totalOffspring = totalPairs * 2;
        int totalGenes = tsp.offspringFlat.size(); // = totalOffspring * tsp.numCities
        // 将 TSP.offspringFlat 拷贝到 device（如果之前没有保持在 device 上）
        cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        unsigned long seed = time(nullptr);
        mutationKernel<<<blocks, threads>>>(tsp.d_offspring, totalOffspring, tsp.numCities, tsp.mutationProbability, seed);
        cudaDeviceSynchronize();
        // 将变异结果拷贝回 TSP.offspringFlat
        cudaMemcpy(tsp.offspringFlat.data(), tsp.d_offspring, totalGenes * sizeof(int), cudaMemcpyDeviceToHost);
        // 更新 Offspring 结构：直接写回，不重新展平
        int offset = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                for (int j = 0; j < tsp.numCities; j++) {
                    child.chromosome[j] = tsp.offspringFlat[offset * tsp.numCities + j];
                }
                offset++;
            }
        }
    }

    // -----------------------------------------------------------
    // updateOffspringFitnessCUDA
    // 使用 TSP.offspringFlat 中已有展平数据
    // -----------------------------------------------------------
    void updateOffspringFitnessCUDA(TSP &tsp, Offspring &offspring) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int totalOffspring = totalPairs * 2;
        int totalGenes = tsp.offspringFlat.size();
        // 直接将 TSP.offspringFlat 拷贝到 device
        cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        computeFitnessKernel<<<blocks, threads>>>(tsp.d_offspring, tsp.d_distanceMatrix, tsp.d_offspringFitness, tsp.numCities, totalOffspring);
        cudaDeviceSynchronize();
        tsp.offspringFitnessFlat.resize(totalOffspring);
        cudaMemcpy(tsp.offspringFitnessFlat.data(), tsp.d_offspringFitness, totalOffspring * sizeof(float), cudaMemcpyDeviceToHost);
        // 更新 Offspring 结构中的 fitness
        int idx = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                child.fitness = tsp.offspringFitnessFlat[idx++];
            }
        }
    }

    // -----------------------------------------------------------
    // updatePopulationFitnessCUDA
    // -----------------------------------------------------------
    void updatePopulationFitnessCUDA(TSP &tsp) {
        // 确保 tsp.populationFlat 是最新的（TSP 内由外部调用更新）
        // 这里直接拷贝 tsp.populationFlat 到 device
        cudaMemcpy(tsp.d_population, tsp.populationFlat.data(), tsp.populationFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (tsp.popSize + threads - 1) / threads;
        computeFitnessKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_distanceMatrix, tsp.d_populationFitness, tsp.numCities, tsp.popSize);
        cudaDeviceSynchronize();
        std::vector<float> h_fit(tsp.popSize);
        cudaMemcpy(h_fit.data(), tsp.d_populationFitness, tsp.popSize * sizeof(float), cudaMemcpyDeviceToHost);
        int idx = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                ind.fitness = h_fit[idx++];
            }
        }
    }

    // -----------------------------------------------------------
    // replacementCUDA
    // GPU 实现 replacement，选择 child1, child2, pa, pb 中最好的两个替代原父代
    // -----------------------------------------------------------
    void replacementCUDA(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += parentPairs[i].size();
        }
        // 假设 TSP.parentAFlat、parentBFlat 已经更新（由 selectionCUDA 完成）
        // 并且 TSP.offspringFlat、offspringFitnessFlat 已由 crossoverCUDA 和 updateOffspringFitnessCUDA 更新

        // 拷贝父代与子代数据到 device（这些展平数组已存在于 TSP 对象中）
        cudaMemcpy(tsp.d_parentA, tsp.parentAFlat.data(), tsp.parentAFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_parentB, tsp.parentBFlat.data(), tsp.parentBFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_parentFitness, tsp.parentFitnessFlat.data(), tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
        // 子代：child1 和 child2分别存储在 TSP.offspringFlat 的前后半部分
        int totalGenesPerChild = totalPairs * tsp.numCities;
        cudaMemcpy(tsp.d_child1, tsp.offspringFlat.data(), totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_child2, tsp.offspringFlat.data() + totalGenesPerChild, totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
        // offspring fitness 已经在 TSP.offspringFitnessFlat 中
        cudaMemcpy(tsp.d_offspringFitness, tsp.offspringFitnessFlat.data(), tsp.offspringFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        replacementKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_populationFitness,
                                                 tsp.d_parentA, tsp.d_parentB, tsp.d_parentFitness,
                                                 tsp.d_child1, tsp.d_child2, tsp.d_offspringFitness,
                                                 totalPairs, tsp.numCities);
        cudaDeviceSynchronize();

        // 将更新后的种群展平数据从 device 拷贝回 TSP.populationFlat
        cudaMemcpy(tsp.populationFlat.data(), tsp.d_population, tsp.populationFlat.size() * sizeof(int), cudaMemcpyDeviceToHost);
        // 同时将更新后的适应度拷贝到 tsp.parentFitnessFlat（用于下一步更新 population 对象）
        cudaMemcpy(tsp.parentFitnessFlat.data(), tsp.d_populationFitness, tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // 更新 TSP.population（假设 TSP.populationFlat 与 population 中的个体顺序一致）
        int offset = 0;
        int fit_idx = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                for (int j = 0; j < tsp.numCities; j++) {
                    ind.chromosome[j] = tsp.populationFlat[offset + j];
                }
                ind.fitness = tsp.parentFitnessFlat[fit_idx++];
                offset += tsp.numCities;
            }
        }
    }

    void migrationCUDA(TSP &tsp) {
        migrationCPU(tsp);
    }
}
