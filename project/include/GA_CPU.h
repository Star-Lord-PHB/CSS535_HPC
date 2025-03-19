#ifndef GA_CPU_H
#define GA_CPU_H

#include "GAInterface.h"

namespace GA {
    ParentPairs selectionCPU(TSP &tsp);
    Offspring crossoverCPU(const TSP &tsp, const ParentPairs &parentPairs);
    void mutationCPU(const TSP &tsp, Offspring &offspring);
    void replacementCPU(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring);
    void migrationCPU(TSP &tsp);

    // CPU 版本适应度计算函数（单个个体）
    float computeFitnessCPU(const Individual &ind, const TSP &tsp);

    // CPU 版本更新整个种群适应度
    void updateFitnessCPU(TSP &tsp);
}

#endif
