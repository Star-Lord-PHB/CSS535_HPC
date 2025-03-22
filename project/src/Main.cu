#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
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
    CUDA_UPDATE_ALL_FITNESS // 全部使用 CUDA 更新适应度
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

// 扩展的 GAResult 结构体，增加各阶段计时信息
struct GAResult {
    std::string strategyName;
    double bestFitness;
    double totalTime; // 主循环迭代总时间
    // 超参数
    int numCities;
    int popSize;
    int mapSize;
    int numIslands;
    float parentSelectionRate;
    float crossoverProbability;
    float mutationProbability;
    int generations;
    // 各阶段时间（单位秒）
    double sel_compute, sel_kernel, sel_total;
    double cross_compute, cross_kernel, cross_total;
    double mut_compute, mut_kernel, mut_total;
    double repl_compute, repl_kernel, repl_total;
    double mig_compute, mig_kernel, mig_total;
    double updPop_compute, updPop_kernel, updPop_total;
    double updOff_compute, updOff_kernel, updOff_total;
};

// 获取策略名称
std::string getStrategyName(Implementation impl) {
    switch (impl) {
        case Implementation::CPU: return "CPU";
        case Implementation::CUDA: return "CUDA";
        case Implementation::CUDA_SELECTION: return "CUDA_SELECTION";
        case Implementation::CUDA_CROSS: return "CUDA_CROSS";
        case Implementation::CUDA_MUTATION: return "CUDA_MUTATION";
        case Implementation::CUDA_REPLACEMENT: return "CUDA_REPLACEMENT";
        case Implementation::CUDA_MIGRATION: return "CUDA_MIGRATION";
        case Implementation::CUDA_UPDATE_POPULATION: return "CUDA_UPDATE_POPULATION";
        case Implementation::CUDA_UPDATE_OFFSPRING: return "CUDA_UPDATE_OFFSPRING";
        case Implementation::CUDA_UPDATE_ALL_FITNESS: return "CUDA_UPDATE_ALL_FITNESS";
        default: return "Unknown";
    }
}

// 封装单个策略的运行过程，返回 GAResult
GAResult run_strategy(Implementation impl,
                      int numCities,
                      int popSize,
                      int mapSize,
                      int numIslands,
                      float parentSelectionRate,
                      float crossoverProbability,
                      float mutationProbability,
                      int generations)
{
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
            gaFuncs.selection = GA::selectionCUDA;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_CROSS:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCUDA;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_MUTATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCUDA;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_REPLACEMENT:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCUDA;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_MIGRATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCUDA;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_UPDATE_POPULATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_UPDATE_OFFSPRING:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
            break;
        case Implementation::CUDA_UPDATE_ALL_FITNESS:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
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

    // 创建 TSP 实例（构造函数中会初始化城市、种群和距离矩阵，同时可能初始化 GPU 数据及计时变量）
    TSP tsp(numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability);

    // 初始适应度更新
    gaFuncs.updatePopulationFitness(tsp);

    // 记录 GA 迭代开始时间（包括所有阶段的执行时间）
    auto startTime = std::chrono::high_resolution_clock::now();

    // GA 主循环
    for (int gen = 0; gen < generations; gen++) {
        gaFuncs.selection(tsp);
        gaFuncs.crossover(tsp);
        gaFuncs.mutation(tsp);
        gaFuncs.updateOffspringFitness(tsp);
        gaFuncs.replacement(tsp);
        gaFuncs.migration(tsp);
        gaFuncs.updatePopulationFitness(tsp);
    }

    // 记录结束时间
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;

    // 遍历种群，找出最佳适应度
    double bestFitness = -1.0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (const auto &ind : tsp.population[island]) {
            if (ind.fitness > bestFitness)
                bestFitness = ind.fitness;
        }
    }

    // 构造结果，包括各阶段的计时数据
    GAResult result;
    result.strategyName = getStrategyName(impl);
    result.bestFitness = bestFitness;
    result.totalTime = duration.count();
    result.numCities = numCities;
    result.popSize = popSize;
    result.mapSize = mapSize;
    result.numIslands = numIslands;
    result.parentSelectionRate = parentSelectionRate;
    result.crossoverProbability = crossoverProbability;
    result.mutationProbability = mutationProbability;
    result.generations = generations;
    // 将各阶段的计时数据复制到 result 中
    result.sel_compute = tsp.selectionTime.computeTime;
    result.sel_kernel  = tsp.selectionTime.kernelTime;
    result.sel_total   = tsp.selectionTime.totalTime;

    result.cross_compute = tsp.crossoverTime.computeTime;
    result.cross_kernel  = tsp.crossoverTime.kernelTime;
    result.cross_total   = tsp.crossoverTime.totalTime;

    result.mut_compute = tsp.mutationTime.computeTime;
    result.mut_kernel  = tsp.mutationTime.kernelTime;
    result.mut_total   = tsp.mutationTime.totalTime;

    result.repl_compute = tsp.replacementTime.computeTime;
    result.repl_kernel  = tsp.replacementTime.kernelTime;
    result.repl_total   = tsp.replacementTime.totalTime;

    result.mig_compute = tsp.migrationTime.computeTime;
    result.mig_kernel  = tsp.migrationTime.kernelTime;
    result.mig_total   = tsp.migrationTime.totalTime;

    result.updPop_compute = tsp.updatePopulationFitnessTime.computeTime;
    result.updPop_kernel  = tsp.updatePopulationFitnessTime.kernelTime;
    result.updPop_total   = tsp.updatePopulationFitnessTime.totalTime;

    result.updOff_compute = tsp.updateOffspringFitnessTime.computeTime;
    result.updOff_kernel  = tsp.updateOffspringFitnessTime.kernelTime;
    result.updOff_total   = tsp.updateOffspringFitnessTime.totalTime;

    return result;
}

int main() {
    // 超参数
    int numCities = 256;
    int popSize = 1024;
    int mapSize = 1000;
    int numIslands = 64;
    float parentSelectionRate = 0.5f;
    float crossoverProbability = 0.7f;
    float mutationProbability = 0.05f;
    int generations = 500;

    // 策略列表
    std::vector<Implementation> strategies = {
        Implementation::CPU,
        Implementation::CUDA,
        Implementation::CUDA_SELECTION,
        Implementation::CUDA_CROSS,
        Implementation::CUDA_MUTATION,
        Implementation::CUDA_REPLACEMENT,
        Implementation::CUDA_MIGRATION,
        Implementation::CUDA_UPDATE_POPULATION,
        Implementation::CUDA_UPDATE_OFFSPRING,
        Implementation::CUDA_UPDATE_ALL_FITNESS
    };

    // 打开 CSV 文件写入数据
    std::ofstream outfile("D://CSS535//CSS535_HPC//project//data.csv");
    outfile << "strategy,best_fitness,total_time,numCities,popSize,mapSize,numIslands,parentSelectionRate,crossoverProbability,mutationProbability,generations,"
            << "sel_compute,sel_kernel,sel_total,"
            << "cross_compute,cross_kernel,cross_total,"
            << "mut_compute,mut_kernel,mut_total,"
            << "repl_compute,repl_kernel,repl_total,"
            << "mig_compute,mig_kernel,mig_total,"
            << "updPop_compute,updPop_kernel,updPop_total,"
            << "updOff_compute,updOff_kernel,updOff_total\n";

    // 遍历每个策略，运行 GA 并写入结果
    for (auto impl : strategies) {
        GAResult res = run_strategy(impl, numCities, popSize, mapSize, numIslands,
                                    parentSelectionRate, crossoverProbability, mutationProbability, generations);
        outfile << res.strategyName << ","
                << res.bestFitness << ","
                << res.totalTime << ","
                << res.numCities << ","
                << res.popSize << ","
                << res.mapSize << ","
                << res.numIslands << ","
                << res.parentSelectionRate << ","
                << res.crossoverProbability << ","
                << res.mutationProbability << ","
                << res.generations << ","
                << res.sel_compute << "," << res.sel_kernel << "," << res.sel_total << ","
                << res.cross_compute << "," << res.cross_kernel << "," << res.cross_total << ","
                << res.mut_compute << "," << res.mut_kernel << "," << res.mut_total << ","
                << res.repl_compute << "," << res.repl_kernel << "," << res.repl_total << ","
                << res.mig_compute << "," << res.mig_kernel << "," << res.mig_total << ","
                << res.updPop_compute << "," << res.updPop_kernel << "," << res.updPop_total << ","
                << res.updOff_compute << "," << res.updOff_kernel << "," << res.updOff_total
                << "\n";
        std::cout << "Finished strategy " << res.strategyName
                  << ": best fitness = " << res.bestFitness
                  << ", total time = " << res.totalTime << " seconds."
                  << std::endl;
    }

    outfile.close();
    return 0;
}
