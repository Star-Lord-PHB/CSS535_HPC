#ifndef GA_CUDA_H
#define GA_CUDA_H

#include "GAInterface.h"

namespace GA {
    ParentPairs selectionCUDA(TSP &tsp);
    Offspring crossoverCUDA(const TSP &tsp, const ParentPairs &parentPairs);
    void mutationCUDA(const TSP &tsp, Offspring &offspring);
    void replacementCUDA(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring);
    void migrationCUDA(TSP &tsp);

    // CUDA 版本适应度计算函数（单个个体版占位）
    float computeFitnessCUDA(const Individual &ind, const TSP &tsp);

    // CUDA 版本更新种群适应度：用于更新父代个体的 fitness
    void updatePopulationFitnessCUDA(TSP &tsp);

    // CUDA 版本更新后代适应度：用于仅更新 offspring 的 fitness
    void updateOffspringFitnessCUDA(TSP &tsp, Offspring &offspring);
}

#endif
