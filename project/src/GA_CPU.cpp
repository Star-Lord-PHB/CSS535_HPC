#include "GA_CPU.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include "TSP.h"

namespace GA {

    using namespace std::chrono;

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
    // 同步更新父代展平数据（不计入计时）
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
    // 同步更新子代展平数据（不计入计时）
    // ---------------------------------------------------------------------
    void syncOffspringFlatten(TSP &tsp) {
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
    // 记录核心逻辑时间（不包含 syncParentFlatten 这部分）
    // ---------------------------------------------------------------------
    void selectionCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();

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
            if (islandPop.size() % 2 == 1) {
                tsp.parentPairs[island].push_back({ islandPop.back(), islandPop.back() });
            }
        }
        tsp.parentPairCount.clear();
        for (int island = 0; island < tsp.numIslands; island++) {
            tsp.parentPairCount.push_back(tsp.parentPairs[island].size());
        }
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        // 记录核心逻辑时间，不包含同步时间
        tsp.selectionTime.computeTime += coreTime;
        tsp.selectionTime.kernelTime += 0; // CPU无内核调用
        // 同步部分另行测量，不计入 computeTime
        auto t_sync0 = high_resolution_clock::now();
        syncParentFlatten(tsp);
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.selectionTime.totalTime += coreTime + syncTime;
    }

    // ---------------------------------------------------------------------
    // 2) Crossover (CPU)
    // 记录核心逻辑时间（不包含 syncOffspringFlatten 这部分）
    // ---------------------------------------------------------------------
    void crossoverCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();

        tsp.offsprings.clear();
        tsp.offsprings.resize(tsp.numIslands);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        std::uniform_int_distribution<int> pointDist(0, tsp.numCities - 1);

        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &pair : tsp.parentPairs[island]) {
                const Individual &pa = pair.first;
                const Individual &pb = pair.second;
                Individual child1 = pa, child2 = pb;
                if (probDist(rng) < tsp.crossoverProbability) {
                    int point1 = pointDist(rng);
                    int point2 = pointDist(rng);
                    if (point1 > point2) std::swap(point1, point2);
                    std::vector<int> ch1(tsp.numCities, -1), ch2(tsp.numCities, -1);
                    for (int k = point1; k <= point2; k++) {
                        ch1[k] = pa.chromosome[k];
                        ch2[k] = pb.chromosome[k];
                    }
                    int index = (point2 + 1) % tsp.numCities;
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pb.chromosome[idx];
                        if (std::find(ch1.begin() + point1, ch1.begin() + point2 + 1, gene) == ch1.begin() + point2 + 1) {
                            ch1[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    index = (point2 + 1) % tsp.numCities;
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
                child1.fitness = 0.0f;
                child2.fitness = 0.0f;
                child1.islandID = pa.islandID;
                child2.islandID = pa.islandID;
                tsp.offsprings[island].push_back(child1);
                tsp.offsprings[island].push_back(child2);
            }
        }
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        tsp.crossoverTime.computeTime += coreTime;
        tsp.crossoverTime.kernelTime += 0;
        // 同步部分：不计入 coreTime
        auto t_sync0 = high_resolution_clock::now();
        syncOffspringFlatten(tsp);
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.crossoverTime.totalTime += coreTime + syncTime;
    }

    // ---------------------------------------------------------------------
    // 3) Mutation (CPU)
    // 记录核心逻辑时间（不包含 syncOffspringFlatten 这部分）
    // ---------------------------------------------------------------------
    void mutationCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();

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
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        tsp.mutationTime.computeTime += coreTime;
        tsp.mutationTime.kernelTime += 0;
        auto t_sync0 = high_resolution_clock::now();
        syncOffspringFlatten(tsp);
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.mutationTime.totalTime += coreTime + syncTime;
    }

    // ---------------------------------------------------------------------
    // 4) Replacement (CPU)
    // 记录核心逻辑时间（不包含 flattenPopulationToHost 这部分）
    // ---------------------------------------------------------------------
    void replacementCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();

        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < tsp.parentPairs[island].size(); i++) {
                Individual candidates[4] = {
                    tsp.parentPairs[island][i].first,
                    tsp.parentPairs[island][i].second,
                    tsp.offsprings[island][2 * i],
                    tsp.offsprings[island][2 * i + 1]
                };
                std::sort(candidates, candidates + 4, [](const Individual &a, const Individual &b) {
                    return a.fitness > b.fitness;
                });
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
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        tsp.replacementTime.computeTime += coreTime;
        tsp.replacementTime.kernelTime += 0;
        auto t_sync0 = high_resolution_clock::now();
        tsp.flattenPopulationToHost();
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.replacementTime.totalTime += coreTime + syncTime;
    }

    // ---------------------------------------------------------------------
    // 5) Migration (CPU)
    // 记录核心逻辑时间（不包含 flattenPopulationToHost 这部分）
    // ---------------------------------------------------------------------
    void migrationCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();

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
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        tsp.migrationTime.computeTime += coreTime;
        tsp.migrationTime.kernelTime += 0;
        auto t_sync0 = high_resolution_clock::now();
        tsp.flattenPopulationToHost();
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.migrationTime.totalTime += coreTime + syncTime;
    }

    // ---------------------------------------------------------------------
    // 6) Update Population Fitness (CPU)
    // 记录核心逻辑时间（不包含 flattenPopulationToHost 这部分）
    // ---------------------------------------------------------------------
    void updatePopulationFitnessCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                ind.fitness = computeFitnessCPU(ind, tsp);
            }
        }
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        tsp.updatePopulationFitnessTime.computeTime += coreTime;
        tsp.updatePopulationFitnessTime.kernelTime += 0;
        auto t_sync0 = high_resolution_clock::now();
        tsp.flattenPopulationToHost();
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.updatePopulationFitnessTime.totalTime += coreTime + syncTime;
    }

    // ---------------------------------------------------------------------
    // 7) Update Offspring Fitness (CPU)
    // 记录核心逻辑时间（不包含 syncOffspringFlatten 这部分）
    // ---------------------------------------------------------------------
    void updateOffspringFitnessCPU(TSP &tsp) {
        auto t0 = high_resolution_clock::now();
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : tsp.offsprings[island]) {
                child.fitness = computeFitnessCPU(child, tsp);
            }
        }
        auto t1 = high_resolution_clock::now();
        double coreTime = duration_cast<duration<double>>(t1 - t0).count();
        tsp.updateOffspringFitnessTime.computeTime += coreTime;
        tsp.updateOffspringFitnessTime.kernelTime += 0;
        auto t_sync0 = high_resolution_clock::now();
        syncOffspringFlatten(tsp);
        auto t_sync1 = high_resolution_clock::now();
        double syncTime = duration_cast<duration<double>>(t_sync1 - t_sync0).count();
        tsp.updateOffspringFitnessTime.totalTime =+ coreTime + syncTime;
    }

} // namespace GA
