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

    // CUDA 版本更新整个种群适应度的函数（真正利用 CUDA 内核批量更新）
    void updateFitnessCUDA(TSP &tsp);
}

#endif
