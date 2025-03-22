#include "GA_CPU.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "TSP.h"

namespace GA {

    // ---------------------------------------------------------------------
    // 计算单个个体适应度（CPU版）
    // 适应度 = 1/(总路径距离)
    // ---------------------------------------------------------------------
    float computeFitnessCPU(const Individual &ind, const TSP &tsp) {
        float totalDistance = 0.0f;
        for (int i = 0; i < tsp.numCities - 1; i++) {
            int city1 = ind.chromosome[i];
            int city2 = ind.chromosome[i + 1];
            totalDistance += tsp.distanceMatrix[city1][city2];
        }
        // 返回起点的距离
        totalDistance += tsp.distanceMatrix[ind.chromosome[tsp.numCities - 1]][ind.chromosome[0]];
        return (totalDistance <= 0.0f) ? 0.0f : 1.0f / totalDistance;
    }

    // ---------------------------------------------------------------------
    // 同步更新父代展平数据
    // 遍历 tsp.parentPairs，将每对父代的染色体和适应度写入展平数组
    // ---------------------------------------------------------------------
    void syncParentFlatten(TSP &tsp) {
        int totalPairs = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            totalPairs += tsp.parentPairs[island].size();
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
                tsp.parentFitnessFlat[2 * pairIndex]     = pa.fitness;
                tsp.parentFitnessFlat[2 * pairIndex + 1] = pb.fitness;
                pairIndex++;
            }
        }
    }

    // ---------------------------------------------------------------------
    // 同步更新子代展平数据
    // 遍历 tsp.offsprings，将每个子代的染色体和适应度写入展平数组
    // ---------------------------------------------------------------------
    void syncOffspringFlatten(TSP &tsp) {
        // std::cout<<"entering syncOffspringFlatten"<<std::endl;
        int totalOffspring = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            totalOffspring += tsp.offsprings[island].size();
        }
        tsp.offspringFlat.resize(totalOffspring * tsp.numCities);
        tsp.offspringFitnessFlat.resize(totalOffspring);

        int offset = 0; // 在 offspringFlat 中的起始索引
        int childIndex = 0; // 子代计数
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < tsp.offsprings[island].size(); i++) {
                const Individual &child = tsp.offsprings[island][i];
                for (int j = 0; j < tsp.numCities; j++) {
                    tsp.offspringFlat[offset + j] = child.chromosome[j];
                }
                tsp.offspringFitnessFlat[childIndex] = child.fitness;
                offset += tsp.numCities;
                childIndex++;
            }
        }
    }

    // ---------------------------------------------------------------------
    // 1) Selection (CPU)
    // 对每个岛内的个体随机洗牌、成对配对，更新 tsp.parentPairs 及 tsp.parentPairCount，
    // 最后同步更新展平的父代数据（parentAFlat、parentBFlat、parentFitnessFlat）
    // ---------------------------------------------------------------------
    void selectionCPU(TSP &tsp) {
        tsp.parentPairs.clear();
        tsp.parentPairs.resize(tsp.numIslands);

        std::mt19937 rng(std::random_device{}());
        for (int island = 0; island < tsp.numIslands; island++) {
            auto &islandPop = tsp.population[island];
            std::shuffle(islandPop.begin(), islandPop.end(), rng);
            int numPairs = islandPop.size() / 2;
            for (int i = 0; i < numPairs; i++) {
                tsp.parentPairs[island].push_back({ islandPop[2 * i], islandPop[2 * i + 1] });
            }
            // 如果岛上个体数为奇数，将最后一个与自身配对
            if (islandPop.size() % 2 == 1) {
                tsp.parentPairs[island].push_back({ islandPop.back(), islandPop.back() });
            }
        }
        // 更新每个岛的配对数统计
        tsp.parentPairCount.clear();
        for (int island = 0; island < tsp.numIslands; island++) {
            tsp.parentPairCount.push_back(tsp.parentPairs[island].size());
        }
        // 同步更新展平的父代数据
        syncParentFlatten(tsp);
    }

    // ---------------------------------------------------------------------
    // 2) Crossover (CPU)
    // 根据 tsp.parentPairs 进行顺序交叉（OX），生成子代存入 tsp.offsprings，
    // 最后同步更新展平的子代数据（offspringFlat、offspringFitnessFlat）
    // ---------------------------------------------------------------------
    void crossoverCPU(TSP &tsp) {
        tsp.offsprings.clear();
        tsp.offsprings.resize(tsp.numIslands);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        std::uniform_int_distribution<int> pointDist(0, tsp.numCities - 1);

        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &pair : tsp.parentPairs[island]) {
                const Individual &pa = pair.first;
                const Individual &pb = pair.second;
                // 以父代拷贝初始化子代
                Individual child1 = pa, child2 = pb;
                if (probDist(rng) < tsp.crossoverProbability) {
                    int point1 = pointDist(rng);
                    int point2 = pointDist(rng);
                    if (point1 > point2) std::swap(point1, point2);
                    std::vector<int> ch1(tsp.numCities, -1), ch2(tsp.numCities, -1);
                    // 复制交叉区间
                    for (int k = point1; k <= point2; k++) {
                        ch1[k] = pa.chromosome[k];
                        ch2[k] = pb.chromosome[k];
                    }
                    int index = (point2 + 1) % tsp.numCities;
                    // 填充 child1：从 pb 中依次取未出现的基因
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pb.chromosome[idx];
                        if (std::find(ch1.begin() + point1, ch1.begin() + point2 + 1, gene) == ch1.begin() + point2 + 1) {
                            ch1[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    index = (point2 + 1) % tsp.numCities;
                    // 填充 child2：从 pa 中依次取未出现的基因
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pa.chromosome[idx];
                        if (std::find(ch2.begin() + point1, ch2.begin() + point2 + 1, gene) == ch2.begin() + point2 + 1) {
                            ch2[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    child1.chromosome = ch1;
                    child2.chromosome = ch2;
                }
                // 重置子代适应度（待后续更新）
                child1.fitness = 0.0f;
                child2.fitness = 0.0f;
                child1.islandID = pa.islandID;
                child2.islandID = pa.islandID;
                tsp.offsprings[island].push_back(child1);
                tsp.offsprings[island].push_back(child2);
            }
        }
        // 同步更新展平的子代数据
        syncOffspringFlatten(tsp);
    }

    // ---------------------------------------------------------------------
    // 3) Mutation (CPU)
    // 对 tsp.offsprings 中的每个子代进行变异（随机交换基因），
    // 变异后同步更新展平的子代数据
    // ---------------------------------------------------------------------
    void mutationCPU(TSP &tsp) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);

        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : tsp.offsprings[island]) {
                for (int i = 0; i < tsp.numCities; i++) {
                    if (probDist(rng) < tsp.mutationProbability) {
                        std::uniform_int_distribution<int> indexDist(0, tsp.numCities - 1);
                        int j = indexDist(rng);
                        std::swap(child.chromosome[i], child.chromosome[j]);
                    }
                }
            }
        }
        // 同步更新展平的子代数据
        syncOffspringFlatten(tsp);
    }

    // ---------------------------------------------------------------------
    // 4) Replacement (CPU)
    // 对于每个父代配对及对应的两个子代，从 {父代A, 父代B, 子代1, 子代2} 中选择适应度最高的两个，
    // 并用它们替换种群中原来的父代；替换后调用 flattenPopulationToHost 同步更新 populationFlat
    // ---------------------------------------------------------------------
    void replacementCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < tsp.parentPairs[island].size(); i++) {
                // 构造四个候选个体
                Individual candidates[4] = {
                    tsp.parentPairs[island][i].first,
                    tsp.parentPairs[island][i].second,
                    tsp.offsprings[island][2 * i],
                    tsp.offsprings[island][2 * i + 1]
                };
                std::sort(candidates, candidates + 4, [](const Individual &a, const Individual &b) {
                    return a.fitness > b.fitness;
                });
                // 在种群中找到与这对父代对应的个体，并替换为排名最高的两个
                auto &pop = tsp.population[island];
                for (auto &ind : pop) {
                    if (ind.chromosome == tsp.parentPairs[island][i].first.chromosome) {
                        ind = candidates[0];
                    } else if (ind.chromosome == tsp.parentPairs[island][i].second.chromosome) {
                        ind = candidates[1];
                    }
                }
            }
        }
        // 同步更新种群展平数据
        tsp.flattenPopulationToHost();
    }

    // ---------------------------------------------------------------------
    // 5) Migration (CPU)
    // 环形迁移：对每个岛，找出最佳和最差个体，用前一岛的最佳替换当前岛的最差（若更优），
    // 更新后同步更新 populationFlat
    // ---------------------------------------------------------------------
    void migrationCPU(TSP &tsp) {
        int nIslands = tsp.numIslands;
        std::vector<Individual> bestInds(nIslands);
        std::vector<int> worstIndex(nIslands, -1);

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
        for (int island = 0; island < nIslands; island++) {
            int prev = (island - 1 + nIslands) % nIslands;
            if (bestInds[prev].fitness > tsp.population[island][worstIndex[island]].fitness) {
                tsp.population[island][worstIndex[island]] = bestInds[prev];
            }
        }
        // 同步更新种群展平数据
        tsp.flattenPopulationToHost();
    }

    // ---------------------------------------------------------------------
    // 6) Update Population Fitness (CPU)
    // 遍历 tsp.population，更新每个个体的适应度，更新后同步更新 populationFlat
    // ---------------------------------------------------------------------
    void updatePopulationFitnessCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                ind.fitness = computeFitnessCPU(ind, tsp);
            }
        }
        tsp.flattenPopulationToHost();
    }

    // ---------------------------------------------------------------------
    // 7) Update Offspring Fitness (CPU)
    // 遍历 tsp.offsprings，更新每个子代的适应度，更新后同步更新 offspringFlat 与 offspringFitnessFlat
    // ---------------------------------------------------------------------
    void updateOffspringFitnessCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : tsp.offsprings[island]) {
                child.fitness = computeFitnessCPU(child, tsp);
            }
        }
        syncOffspringFlatten(tsp);
    }

} // namespace GA
