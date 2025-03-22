#include <iostream>
#include <chrono>
#include "TSP.h"
#include "GAInterface.h"
#include "GA_CPU.h"
#include "GA_CUDA.h"
#include "GA_oneThreadPerGene.hpp" // Additional GA variant (if needed)

// 扩展的实现策略枚举
enum class Implementation {
    CPU,
    CUDA,
    CUDA_SELECTION,        // 仅 selection 使用 CUDA，其余均用 CPU
    CUDA_CROSS,            // 仅 crossover 使用 CUDA
    CUDA_MUTATION,         // 仅 mutation 使用 CUDA
    CUDA_REPLACEMENT,      // 仅 replacement 使用 CUDA
    CUDA_MIGRATION,        // 仅 migration 使用 CUDA
    CUDA_UPDATE_POPULATION,  // 仅更新 population 适应度使用 CUDA
    CUDA_UPDATE_OFFSPRING,    // 仅更新 offspring 适应度使用 CUDA
    CUDA_UPDATE_ALL_FITNESS // 全部使用 CUDA
};

// 函数集结构体，用于保存 GA 各部分的函数指针
struct GAFunctionSet {
    void (*selection)(TSP &);
    void (*crossover)(TSP &);
    void (*mutation)(TSP &);
    void (*replacement)(TSP &);
    void (*migration)(TSP &);
    void (*updatePopulationFitness)(TSP &);
    void (*updateOffspringFitness)(TSP &);
};

int main() {
    // GA 算法参数
    int numCities = 256;              // 城市数量
    int popSize = 1024;               // 总种群大小（跨所有岛屿）
    int mapSize = 1000;               // 地图尺寸（例如城市坐标范围 0～mapSize）
    int numIslands = 64;              // 岛屿数量（子种群数量）
    float parentSelectionRate = 0.5f; // 父代选择率（可选参数）
    float crossoverProbability = 0.7f;// 交叉概率
    float mutationProbability = 0.05f; // 变异概率
    int generations = 500;            // 运行代数

    // 选择实现策略（可选：CPU、CUDA、或部分环节采用 CUDA）
    // 例如：下面选择 CUDA_SELECTION，即仅 selection 部分使用 CUDA，其余均使用 CPU 实现。
    Implementation impl = Implementation::CUDA_SELECTION;

    // 根据选择的策略设置函数指针
    GAFunctionSet gaFuncs;
    switch (impl) {
        case Implementation::CPU:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA:
            gaFuncs.selection = GA::selectionCUDA;
            gaFuncs.crossover = GA::crossoverCUDA;
            gaFuncs.mutation = GA::mutationCUDA;
            gaFuncs.replacement = GA::replacementCUDA;
            gaFuncs.migration = GA::migrationCUDA;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
            break;
        case Implementation::CUDA_SELECTION:
            gaFuncs.selection = GA::selectionCUDA;   // 采用 CUDA 版本
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_CROSS:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCUDA;     // 采用 CUDA 版本
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_MUTATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCUDA;       // 采用 CUDA 版本
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_REPLACEMENT:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCUDA;   // 采用 CUDA 版本
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_MIGRATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCUDA;       // 采用 CUDA 版本
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_UPDATE_POPULATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA; // 采用 CUDA 版本
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_UPDATE_OFFSPRING:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;   // 采用 CUDA 版本
            break;
        case Implementation::CUDA_UPDATE_ALL_FITNESS:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA; // 采用 CUDA 版本
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;   // 采用 CUDA 版本
            break;
        default:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
    }

    // 初始化 TSP 问题实例（构造函数中会生成城市、种群和距离矩阵）
    TSP tsp(numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability);

    // 初始适应度更新：计算整个种群的适应度
    gaFuncs.updatePopulationFitness(tsp);

    // 记录 GA 迭代开始时间
    auto startTime = std::chrono::high_resolution_clock::now();

    // GA 主循环，每一代进行各步骤的迭代
    for (int gen = 0; gen < generations; gen++) {
        // Selection: 更新 tsp.parentPairs 和 parentPairCount（同步更新对应的展平数据由各函数内部完成）
        gaFuncs.selection(tsp);
        // Crossover: 更新 tsp.offsprings 以及展平的 offspringFlat
        gaFuncs.crossover(tsp);
        // Mutation: 对 tsp.offsprings 进行变异
        gaFuncs.mutation(tsp);
        // Update offspring fitness: 仅对新生成的 offspring 更新适应度
        gaFuncs.updateOffspringFitness(tsp);
        // Replacement: 根据父代和 offspring 的最优个体更新 tsp.population
        gaFuncs.replacement(tsp);
        // Migration: 执行岛间迁移，更新 tsp.population
        gaFuncs.migration(tsp);
        // Update population fitness: 更新整个种群的适应度
        gaFuncs.updatePopulationFitness(tsp);

        std::cout << "Generation " << gen << " complete." << std::endl;
        // for (int island = 0; island < tsp.numIslands; island++) {
        //     float bestFit = -1.0f;
        //     for (const auto &ind : tsp.population[island]) {
        //         if (ind.fitness > bestFit)
        //             bestFit = ind.fitness;
        //     }
        //     std::cout << "  Island " << island << " best fitness: " << bestFit << std::endl;
        // }
    }

    // 记录 GA 迭代结束时间
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "Total GA iterations time: " << duration.count() << " seconds." << std::endl;

    return 0;
}
