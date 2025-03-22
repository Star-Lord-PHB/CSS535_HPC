#include "GA_CUDA.h"
#include "GA_CPU.h"  // 可参考 CPU 版本中的同步函数
#include "TSP.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>
#include <iostream>

namespace GA {

    // ---------------------------------------------------------------------
    // Kernel: Compute fitness for each individual
    // ---------------------------------------------------------------------
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
            int lastCity = d_population[base + numCities - 1];
            int firstCity = d_population[base];
            totalDist += d_distanceMatrix[lastCity * numCities + firstCity];
            d_fitness[idx] = (totalDist <= 0.0f) ? 0.0f : (1.0f / totalDist);
        }
    }

    // ---------------------------------------------------------------------
    // Kernel: Order Crossover (OX) for a pair of parents.
    // ---------------------------------------------------------------------
    __global__ void crossoverKernel(const int *d_parentA, const int *d_parentB,
                                      int *d_child1, int *d_child2,
                                      int numPairs, int numCities,
                                      float crossoverProb, unsigned long seed)
    {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx >= numPairs) return;

        curandState state;
        curand_init(seed, pairIdx, 0, &state);

        int base = pairIdx * numCities;
        int *child1 = d_child1 + base;
        int *child2 = d_child2 + base;

        float r = curand_uniform(&state);
        if (r >= crossoverProb) {
            for (int i = 0; i < numCities; i++) {
                child1[i] = d_parentA[base + i];
                child2[i] = d_parentB[base + i];
            }
            return;
        }

        // Initialize children with -1
        for (int i = 0; i < numCities; i++) {
            child1[i] = -1;
            child2[i] = -1;
        }
        int p1 = curand(&state) % numCities;
        int p2 = curand(&state) % numCities;
        if (p1 > p2) { int tmp = p1; p1 = p2; p2 = tmp; }
        for (int i = p1; i <= p2; i++) {
            child1[i] = d_parentA[base + i];
            child2[i] = d_parentB[base + i];
        }
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

    // ---------------------------------------------------------------------
    // Kernel: Mutation: Each thread processes one individual
    // ---------------------------------------------------------------------
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

    // ---------------------------------------------------------------------
    // Kernel: Replacement
    // Each thread processes one parent pair (with two children).
    // It selects the two best individuals from {parent A, parent B, child1, child2}
    // and writes them back to the flattened population.
    // ---------------------------------------------------------------------
    __global__ void replacementKernel(
        int *d_population, float *d_populationFitness,
        const int *d_parentA, const int *d_parentB,
        const float *d_parentFitness,
        const int *d_child1, const int *d_child2,
        const float *d_childFitness,
        int numPairs, int numCities, int totalPairs)
    {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx >= numPairs) return;

        int base = pairIdx * numCities;
        float fits[4] = {
            d_parentFitness[2 * pairIdx],
            d_parentFitness[2 * pairIdx + 1],
            d_childFitness[pairIdx],
            d_childFitness[totalPairs + pairIdx]
        };
        const int* chrom[4] = {
            d_parentA + base,
            d_parentB + base,
            d_child1 + base,
            d_child2 + base
        };

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
        int popOffsetA = pairIdx * 2 * numCities;      // Parent A position in flattened population
        int popOffsetB = popOffsetA + numCities;         // Parent B position
        for (int i = 0; i < numCities; i++) {
            d_population[popOffsetA + i] = chrom[bestIdx][i];
            d_population[popOffsetB + i] = chrom[secondIdx][i];
        }
        d_populationFitness[2 * pairIdx] = fits[bestIdx];
        d_populationFitness[2 * pairIdx + 1] = fits[secondIdx];
    }

    // ---------------------------------------------------------------------
    // selectionCUDA (同步更新 CPU 数据)
    // 调用 CPU 版本的 selectionCPU 完成 tsp.parentPairs 的更新，
    // 然后将其展平更新到 tsp.parentAFlat、tsp.parentBFlat 和 tsp.parentFitnessFlat
    // ---------------------------------------------------------------------
    void selectionCUDA(TSP &tsp) {
        // 调用 CPU 版本选择函数，更新 tsp.parentPairs
        selectionCPU(tsp);
        // 同步更新父代展平数据
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        tsp.parentAFlat.resize(totalPairs * tsp.numCities);
        tsp.parentBFlat.resize(totalPairs * tsp.numCities);
        tsp.parentFitnessFlat.resize(totalPairs * 2);
        int pairIndex = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < tsp.parentPairs[island].size(); i++) {
                const Individual &pa = tsp.parentPairs[island][i].first;
                const Individual &pb = tsp.parentPairs[island][i].second;
                for (int j = 0; j < tsp.numCities; j++) {
                    tsp.parentAFlat[pairIndex * tsp.numCities + j] = pa.chromosome[j];
                    tsp.parentBFlat[pairIndex * tsp.numCities + j] = pb.chromosome[j];
                }
                tsp.parentFitnessFlat[2 * pairIndex] = pa.fitness;
                tsp.parentFitnessFlat[2 * pairIndex + 1] = pb.fitness;
                pairIndex++;
            }
        }
        // 此时 CPU 端的父代数据（非展平及展平版本）均已更新
    }

    // ---------------------------------------------------------------------
    // crossoverCUDA (同步更新 CPU 端 offspring 数据)
    // 使用展平的 tsp.parentAFlat 与 tsp.parentBFlat 执行交叉，
    // 并重构 tsp.offsprings，同时更新展平的 tsp.offspringFlat
    // ---------------------------------------------------------------------
    void crossoverCUDA(TSP &tsp) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        unsigned long seed = time(nullptr);
        // 启动交叉 kernel
        crossoverKernel<<<blocks, threads>>>(tsp.d_parentA, tsp.d_parentB, tsp.d_child1, tsp.d_child2,
                                               totalPairs, tsp.numCities, tsp.crossoverProbability, seed);
        cudaDeviceSynchronize();
        // 从设备读取交叉结果到展平数组 tsp.offspringFlat
        tsp.offspringFlat.resize(totalPairs * 2 * tsp.numCities);
        cudaMemcpy(tsp.offspringFlat.data(), tsp.d_child1, totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tsp.offspringFlat.data() + totalPairs * tsp.numCities, tsp.d_child2,
                   totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
        // 重构 CPU 端 tsp.offsprings 结构
        Offspring offsprings;
        offsprings.resize(tsp.numIslands);
        int pairIndex = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            int numPairs = tsp.parentPairs[island].size();
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
                offsprings[island].push_back(child1);
                offsprings[island].push_back(child2);
                pairIndex++;
            }
        }
        tsp.offsprings = offsprings;
        // 此时 CPU 端的 offspring（及展平数据）均已更新
    }

    // ---------------------------------------------------------------------
    // mutationCUDA (同步更新 CPU 端 offspring 数据)
    // 使用展平的 tsp.offspringFlat 执行变异，变异后更新 tsp.offsprings
    // ---------------------------------------------------------------------
    void mutationCUDA(TSP &tsp) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int totalOffspring = totalPairs * 2;
        int totalGenes = tsp.offspringFlat.size(); // totalOffspring * tsp.numCities
        cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        unsigned long seed = time(nullptr);
        mutationKernel<<<blocks, threads>>>(tsp.d_offspring, totalOffspring, tsp.numCities, tsp.mutationProbability, seed);
        cudaDeviceSynchronize();
        cudaMemcpy(tsp.offspringFlat.data(), tsp.d_offspring, totalGenes * sizeof(int), cudaMemcpyDeviceToHost);
        // 更新 CPU 端 tsp.offsprings 数据
        int offset = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : tsp.offsprings[island]) {
                for (int j = 0; j < tsp.numCities; j++) {
                    child.chromosome[j] = tsp.offspringFlat[offset * tsp.numCities + j];
                }
                offset++;
            }
        }
        // 同步完成
    }

    // ---------------------------------------------------------------------
    // updateOffspringFitnessCUDA (同步更新 CPU 端 offspring 的适应度)
    // 使用展平的 tsp.offspringFlat 计算适应度，并更新 tsp.offsprings 中各子代的 fitness
    // ---------------------------------------------------------------------
    void updateOffspringFitnessCUDA(TSP &tsp) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int totalOffspring = totalPairs * 2;
        int totalGenes = tsp.offspringFlat.size();
        cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        computeFitnessKernel<<<blocks, threads>>>(tsp.d_offspring, tsp.d_distanceMatrix, tsp.d_offspringFitness, tsp.numCities, totalOffspring);
        cudaDeviceSynchronize();
        tsp.offspringFitnessFlat.resize(totalOffspring);
        cudaMemcpy(tsp.offspringFitnessFlat.data(), tsp.d_offspringFitness, totalOffspring * sizeof(float), cudaMemcpyDeviceToHost);
        int idx = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : tsp.offsprings[island]) {
                child.fitness = tsp.offspringFitnessFlat[idx++];
            }
        }
        // 同步后，CPU 端 offspring 的适应度已更新
    }

    // ---------------------------------------------------------------------
    // updatePopulationFitnessCUDA (同步更新 CPU 端 population 中各个体的适应度)
    // ---------------------------------------------------------------------
    void updatePopulationFitnessCUDA(TSP &tsp) {
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
        // 此处可根据需要调用 tsp.flattenPopulationToHost() 更新展平数据
    }

    // ---------------------------------------------------------------------
    // replacementCUDA (同步更新 CPU 端 population 数据)
    // 执行替换操作后，将更新后的种群展平数据复制回 CPU，并重构 tsp.population
    // ---------------------------------------------------------------------
    void replacementCUDA(TSP &tsp) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        cudaMemcpy(tsp.d_parentA, tsp.parentAFlat.data(), tsp.parentAFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_parentB, tsp.parentBFlat.data(), tsp.parentBFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_parentFitness, tsp.parentFitnessFlat.data(), tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
        int totalGenesPerChild = totalPairs * tsp.numCities;
        cudaMemcpy(tsp.d_child1, tsp.offspringFlat.data(), totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_child2, tsp.offspringFlat.data() + totalGenesPerChild, totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_offspringFitness, tsp.offspringFitnessFlat.data(), tsp.offspringFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        replacementKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_populationFitness,
                                                 tsp.d_parentA, tsp.d_parentB, tsp.d_parentFitness,
                                                 tsp.d_child1, tsp.d_child2, tsp.d_offspringFitness,
                                                 totalPairs, tsp.numCities, totalPairs);
        cudaDeviceSynchronize();

        cudaMemcpy(tsp.populationFlat.data(), tsp.d_population, tsp.populationFlat.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tsp.parentFitnessFlat.data(), tsp.d_populationFitness, tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyDeviceToHost);

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
        // 此时 CPU 端的种群数据已更新同步
    }

    // ---------------------------------------------------------------------
    // migrationCUDA (同步更新 CPU 端 population 数据)
    // 直接调用 CPU 版本的 migrationCPU，然后更新展平数组
    // ---------------------------------------------------------------------
    void migrationCUDA(TSP &tsp) {
        migrationCPU(tsp);
        tsp.flattenPopulationToHost();
        // CPU 端的 tsp.population 和展平数组已同步
    }

} // namespace GA
