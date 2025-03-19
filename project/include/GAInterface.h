#ifndef GA_INTERFACE_H
#define GA_INTERFACE_H

#include "TSP.h"
#include <vector>
#include <utility>

namespace GA {

    // 用于保存每个岛的父代配对，配对的两个 Individual 代表一个父代对
    typedef std::vector<std::vector<std::pair<Individual, Individual>>> ParentPairs;
    // 后代集合：按岛存放，每个岛中是若干 Individual 后代
    typedef std::vector<std::vector<Individual>> Offspring;

    // 各模块接口函数原型
    ParentPairs selection(TSP &tsp);
    Offspring crossover(const TSP &tsp, const ParentPairs &parentPairs);
    void mutation(const TSP &tsp, Offspring &offspring);
    void replacement(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring);
    void migration(TSP &tsp);

    // 新增：更新整个种群适应度的函数
    void updateFitness(TSP &tsp);

} // namespace GA

#endif
