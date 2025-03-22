#include <iostream>
#include <chrono>
#include "TSP.h"
#include "GAInterface.h"
#include "GA_CPU.h"
#include "GA_CUDA.h"
#include "GA_oneThreadPerGene.hpp"
#include  <limits>

using namespace std::chrono;

// 枚举用于选择实现版本
enum class Implementation { CPU, CUDA_perIndividual, CUDA_perGene };


struct GAFunctionSet {
    ParentPairs (*selection)(TSP &);
    Offspring (*crossover)(TSP &, const ParentPairs &);
    void (*mutation)(TSP &, Offspring &);
    void (*replacement)(TSP &, const ParentPairs &, const Offspring &);
    void (*migration)(TSP &);
    void (*updatePopulationFitness)(TSP &);
    void (*updateOffspringFitness)(TSP &, Offspring &);
};


void runOneIndividualPerThread(const GAFunctionSet& gaFuncs, TSP& tsp, const int generations) {

    // 初始适应度更新：更新种群的 fitness
    gaFuncs.updatePopulationFitness(tsp);

    // 记录迭代开始时间
    auto startTime = std::chrono::high_resolution_clock::now();

    // 迭代 GA 算法
    for (int gen = 0; gen < generations; gen++) {
        auto parentPairs = gaFuncs.selection(tsp);
        auto offspring = gaFuncs.crossover(tsp, parentPairs);
        gaFuncs.mutation(tsp, offspring);

        // 更新 offspring 的适应度：只计算新生成的子代
        gaFuncs.updateOffspringFitness(tsp, offspring);

        // 用 offspring 替换种群中较差的父代
        gaFuncs.replacement(tsp, parentPairs, offspring);

        // 岛间迁移
        gaFuncs.migration(tsp);

        // 替换后更新整个种群的适应度
        gaFuncs.updatePopulationFitness(tsp);

        std::cout << "Generation " << gen << " complete.\n";
        for (int island = 0; island < tsp.numIslands; island++) {
            float bestFit = -1.0f;
            for (const auto &ind : tsp.population[island]) {
                if (ind.fitness > bestFit)
                    bestFit = ind.fitness;
            }
            // std::cout << "  Island " << island << " best fitness: " << bestFit << std::endl;
        }
    }

    // 记录迭代结束时间
    auto endTime = std::chrono::high_resolution_clock::now();
    // 计算迭代部分的持续时间，单位为秒
    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "Total GA iterations time: " << duration.count() << " seconds." << std::endl;

}


void runOneGenePerThread(TSP& tsp, const int generations) {

    auto const solution = new Array3D<int>(tsp.numIslands, tsp.popSize / tsp.numIslands, tsp.numCities + 1);
    auto const distanceMat = new Array2D<float>(tsp.numCities, tsp.numCities);
    distanceMat->fill(tsp.distanceMatrixFlat);

    auto calculationTime = milliseconds();
    auto totalTime = milliseconds();

    ga_one_thread_per_gene(
        *solution,
        *distanceMat,
        generations, tsp.crossoverProbability, tsp.mutationProbability,
        &calculationTime,
        &totalTime
    );

    unsigned bestIslandId = 0;
    unsigned bestIndividualId = 0;
    float bestFitness = INFINITY;
    for (unsigned islandId = 0; islandId < solution->xSize(); islandId++) {
        for (unsigned individualId = 0; individualId < solution->ySize(); individualId++) {
            const auto fitness = *reinterpret_cast<float*>(&solution->at(islandId, individualId, tsp.numCities));
            if (fitness < bestFitness) {
                bestIslandId = islandId;
                bestIndividualId = individualId;
                bestFitness = fitness;
            }
        }
    }

    const auto bestIndividual = std::vector(
        &solution->at(bestIslandId, bestIndividualId, 0),
        &solution->at(bestIslandId, bestIndividualId, tsp.numCities - 1)
    );

    std::cout << "One Gene Per Thread (CUDA) Result:" << std::endl;
    std::cout << "Time Total: " << (calculationTime / 1000).count() << " seconds" << std::endl;
    std::cout << "Best Fitness: " << bestFitness << std::endl;
    std::cout << "Best Individual: ";
    for (const auto& gene : bestIndividual) {
        std::cout << gene << " ";
    }
    std::cout << std::endl;

    delete solution;

}


int main() {
    // GA 算法参数设置
    int numCities = 256;
    int popSize = 1024;
    int mapSize = 1000;
    int numIslands = 64;
    float parentSelectionRate = 0.5f;
    float crossoverProbability = 0.7f;
    float mutationProbability = 0.05f;
    int generations = 500;

    // 选择实现版本
    Implementation impl = Implementation::CUDA_perGene; // CUDA 或 CPU

    // 初始化 TSP 问题（构造函数会生成城市、种群和距离矩阵）
    TSP tsp(numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability);

    auto gaFuncs = GAFunctionSet();
    if (impl == Implementation::CPU) {
        gaFuncs.selection = GA::selectionCPU;
        gaFuncs.crossover = GA::crossoverCPU;
        gaFuncs.mutation = GA::mutationCPU;
        gaFuncs.replacement = GA::replacementCPU;
        gaFuncs.migration = GA::migrationCPU;
        gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
        gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
        runOneIndividualPerThread(gaFuncs, tsp, generations);
    } else if (impl == Implementation::CUDA_perIndividual) { // Implementation::CUDA
        gaFuncs.selection = GA::selectionCUDA;
        gaFuncs.crossover = GA::crossoverCUDA;
        gaFuncs.mutation = GA::mutationCUDA;
        gaFuncs.replacement = GA::replacementCUDA;
        gaFuncs.migration = GA::migrationCUDA;
        gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
        gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
        runOneIndividualPerThread(gaFuncs, tsp, generations);
    } else if (impl == Implementation::CUDA_perGene) {
        runOneGenePerThread(tsp, generations);
    }

    return 0;

}