#include "GA_CPU.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include "TSP.h"

namespace GA {

    // ------------------------------
    // 帮助函数：CPU计算单个个体适应度
    // ------------------------------
    static float computeFitnessCPU(const Individual &ind, const TSP &tsp) {
        float totalDistance = 0.0f;
        for (int i = 0; i < tsp.numCities - 1; i++) {
            int city1 = ind.chromosome[i];
            int city2 = ind.chromosome[i + 1];
            totalDistance += tsp.distanceMatrix[city1][city2];
        }
        // 回到起点
        totalDistance += tsp.distanceMatrix[ind.chromosome[tsp.numCities - 1]][ind.chromosome[0]];
        return (totalDistance <= 0.0f) ? 0.0f : 1.0f / totalDistance;
    }

    // ------------------------------
    // 1) Selection (CPU)
    // ------------------------------
    ParentPairs selectionCPU(TSP &tsp) {
        ParentPairs parentPairs(tsp.numIslands);
        std::mt19937 rng(std::random_device{}());

        for (int island = 0; island < tsp.numIslands; island++) {
            auto &islandPop = tsp.population[island];
            std::shuffle(islandPop.begin(), islandPop.end(), rng);
            int numPairs = islandPop.size() / 2;
            for (int i = 0; i < numPairs; i++) {
                parentPairs[island].push_back({islandPop[2*i], islandPop[2*i + 1]});
            }
            // 若有奇数个体，最后一个自我配对
            if (islandPop.size() % 2 == 1) {
                parentPairs[island].push_back({islandPop.back(), islandPop.back()});
            }
        }
        return parentPairs;
    }

    // ------------------------------
    // 2) Crossover (CPU)
    // ------------------------------
    Offspring crossoverCPU(TSP &tsp, const ParentPairs &parentPairs) {
        Offspring offspring(tsp.numIslands);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        std::uniform_int_distribution<int> pointDist(0, tsp.numCities - 1);

        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &pair : parentPairs[island]) {
                const Individual &pa = pair.first;
                const Individual &pb = pair.second;
                Individual child1 = pa, child2 = pb;

                if (probDist(rng) < tsp.crossoverProbability) {
                    // OX交叉
                    int point1 = pointDist(rng);
                    int point2 = pointDist(rng);
                    if (point1 > point2) std::swap(point1, point2);

                    std::vector<int> ch1(tsp.numCities, -1), ch2(tsp.numCities, -1);
                    for (int k = point1; k <= point2; k++) {
                        ch1[k] = pa.chromosome[k];
                        ch2[k] = pb.chromosome[k];
                    }
                    // 填充 child1
                    int index = (point2 + 1) % tsp.numCities;
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pb.chromosome[idx];
                        if (std::find(ch1.begin(), ch1.end(), gene) == ch1.end()) {
                            ch1[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    // 填充 child2
                    index = (point2 + 1) % tsp.numCities;
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pa.chromosome[idx];
                        if (std::find(ch2.begin(), ch2.end(), gene) == ch2.end()) {
                            ch2[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    child1.chromosome = ch1;
                    child2.chromosome = ch2;
                }
                // 重置子代适应度
                child1.fitness = 0.0f;
                child2.fitness = 0.0f;

                offspring[island].push_back(child1);
                offspring[island].push_back(child2);
            }
        }
        return offspring;
    }

    // ------------------------------
    // 3) Mutation (CPU)
    // ------------------------------
    void mutationCPU(TSP &tsp, Offspring &offspring) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);

        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                for (int i = 0; i < tsp.numCities; i++) {
                    if (probDist(rng) < tsp.mutationProbability) {
                        std::uniform_int_distribution<int> indexDist(0, tsp.numCities - 1);
                        int j = indexDist(rng);
                        std::swap(child.chromosome[i], child.chromosome[j]);
                    }
                }
            }
        }
    }

    // ------------------------------
    // 4) Replacement (CPU)
    // ------------------------------
    void GA::replacementCPU(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < parentPairs[island].size(); i++) {
                Individual candidates[4] = {
                    parentPairs[island][i].first,   // pa
                    parentPairs[island][i].second,  // pb
                    offspring[island][2*i],         // child1
                    offspring[island][2*i + 1]      // child2
                };
                // 排序选出前两名
                std::sort(candidates, candidates + 4,
                          [](const Individual &a, const Individual &b) {
                              return a.fitness > b.fitness; // 降序
                          });

                // 替换原种群中pa和pb
                auto &pop = tsp.population[island];
                for (auto &ind : pop) {
                    if (ind.chromosome == parentPairs[island][i].first.chromosome) {
                        ind = candidates[0];
                    } else if (ind.chromosome == parentPairs[island][i].second.chromosome) {
                        ind = candidates[1];
                    }
                }
            }
        }
        // 更新展平后的种群数据
        tsp.flattenPopulationToHost();
    }


    // ------------------------------
    // 5) Migration (CPU)
    // ------------------------------
    void migrationCPU(TSP &tsp) {
        // 简单的环状迁移示例
        int nIslands = tsp.numIslands;
        std::vector<Individual> bestInds(nIslands);
        std::vector<int> worstIndex(nIslands, -1);

        // 找到每个岛最优个体与最差个体下标
        for (int island = 0; island < nIslands; island++) {
            float bestFit = -1e9, worstFit = 1e9;
            int wIdx = -1;
            for (int i = 0; i < (int)tsp.population[island].size(); i++) {
                float f = tsp.population[island][i].fitness;
                if (f > bestFit) {
                    bestFit = f;
                    bestInds[island] = tsp.population[island][i];
                }
                if (f < worstFit) {
                    worstFit = f;
                    wIdx = i;
                }
            }
            worstIndex[island] = wIdx;
        }
        // 迁移：把上一个岛的最优替换当前岛的最差
        for (int island = 0; island < nIslands; island++) {
            int prev = (island - 1 + nIslands) % nIslands;
            if (bestInds[prev].fitness > tsp.population[island][worstIndex[island]].fitness) {
                tsp.population[island][worstIndex[island]] = bestInds[prev];
            }
        }
        // 同步展平
        tsp.flattenPopulationToHost();
    }

    // ------------------------------
    // 更新种群适应度 (CPU)
    // ------------------------------
    void updatePopulationFitnessCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                ind.fitness = computeFitnessCPU(ind, tsp);
            }
        }
        // 计算完毕后可以同步 populationFlat，无需更新染色体顺序，此处保留
        // 如果某些kernel要用到fitness，也可考虑做额外操作
    }

    // ------------------------------
    // 更新后代适应度 (CPU)
    // ------------------------------
    void updateOffspringFitnessCPU(TSP &tsp, Offspring &offspring) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                child.fitness = computeFitnessCPU(child, tsp);
            }
        }
    }

} // namespace GA
