#include <iostream>
#include "TSP.h"
#include "GAInterface.h"
#include "GA_CPU.h"
#include "GA_CUDA.h"

// 枚举用于选择实现版本
enum class Implementation { CPU, CUDA };

struct GAFunctionSet {
    GA::ParentPairs (*selection)(TSP &);
    GA::Offspring (*crossover)(const TSP &, const GA::ParentPairs &);
    void (*mutation)(const TSP &, GA::Offspring &);
    void (*replacement)(TSP &, const GA::ParentPairs &, const GA::Offspring &);
    void (*migration)(TSP &);
    void (*updatePopulationFitness)(TSP &);
    void (*updateOffspringFitness)(TSP &, GA::Offspring &);
};

int main() {
    // GA 算法参数设置
    int numCities = 10;
    int popSize = 20;
    int mapSize = 100;
    int numIslands = 2;
    float parentSelectionRate = 0.5f;
    float crossoverProbability = 0.7f;
    float mutationProbability = 0.05f;
    int generations = 100;

    // 选择实现版本
    Implementation impl = Implementation::CUDA; // 或 CPU

    GAFunctionSet gaFuncs;
    if (impl == Implementation::CPU) {
        gaFuncs.selection = GA::selectionCPU;
        gaFuncs.crossover = GA::crossoverCPU;
        gaFuncs.mutation = GA::mutationCPU;
        gaFuncs.replacement = GA::replacementCPU;
        gaFuncs.migration = GA::migrationCPU;
        gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
        gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
    } else { // Implementation::CUDA
        gaFuncs.selection = GA::selectionCUDA;
        gaFuncs.crossover = GA::crossoverCUDA;
        gaFuncs.mutation = GA::mutationCUDA;
        gaFuncs.replacement = GA::replacementCUDA;
        gaFuncs.migration = GA::migrationCUDA;
        gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
        gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
    }

    // 初始化 TSP 问题（构造函数会生成城市、种群和距离矩阵）
    TSP tsp(numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability);

    // 初始适应度更新：更新种群的 fitness
    gaFuncs.updatePopulationFitness(tsp);

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
            std::cout << "  Island " << island << " best fitness: " << bestFit << std::endl;
        }
    }

    return 0;
}
