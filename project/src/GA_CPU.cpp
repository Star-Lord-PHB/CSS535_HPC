#include "GA_CPU.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

namespace GA {

// 计算适应度：定义为 1 / (路径总距离)
float computeFitnessCPU(const Individual &ind, const TSP &tsp) {
    float totalDistance = 0.0f;
    for (int i = 0; i < tsp.numCities - 1; i++) {
        int city1 = ind.chromosome[i];
        int city2 = ind.chromosome[i+1];
        totalDistance += tsp.distanceMatrix[city1][city2];
    }
    totalDistance += tsp.distanceMatrix[ind.chromosome[tsp.numCities - 1]][ind.chromosome[0]];
    return (totalDistance <= 0.0f) ? 0.0f : 1.0f / totalDistance;
}

// 更新整个种群适应度：直接遍历各岛与各个个体
void updateFitnessCPU(TSP &tsp) {
    for (int island = 0; island < tsp.numIslands; island++) {
        for (auto &ind : tsp.population[island]) {
            ind.fitness = computeFitnessCPU(ind, tsp);
        }
    }
}

// 1. Selection: 在同一岛内随机配对
ParentPairs selectionCPU(TSP &tsp) {
    ParentPairs parentPairs(tsp.numIslands);
    std::mt19937 rng(std::random_device{}());
    for (int island = 0; island < tsp.numIslands; island++) {
        std::vector<Individual> &islandPop = tsp.population[island];
        std::shuffle(islandPop.begin(), islandPop.end(), rng);
        int numPairs = islandPop.size() / 2;
        for (int i = 0; i < numPairs; i++) {
            parentPairs[island].push_back({islandPop[2*i], islandPop[2*i+1]});
        }
        if (islandPop.size() % 2 == 1) {
            parentPairs[island].push_back({islandPop.back(), islandPop.back()});
        }
    }
    return parentPairs;
}

// 2. Crossover: 顺序交叉
Offspring crossoverCPU(const TSP &tsp, const ParentPairs &parentPairs) {
    Offspring offspring(tsp.numIslands);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    std::uniform_int_distribution<int> pointDist(0, tsp.numCities - 1);

    for (int island = 0; island < tsp.numIslands; island++) {
        for (const auto &pair : parentPairs[island]) {
            const Individual &pa = pair.first;
            const Individual &pb = pair.second;
            Individual child1 = pa, child2 = pb; // 默认直接复制父代

            if (probDist(rng) < tsp.crossoverProbability) {
                // 顺序交叉 (OX) 需要选择两个交叉点
                int point1 = pointDist(rng);
                int point2 = pointDist(rng);
                if (point1 > point2) std::swap(point1, point2);

                // 用 -1 初始化临时染色体，表示该位置尚未填充
                std::vector<int> child1Chromosome(tsp.numCities, -1);
                std::vector<int> child2Chromosome(tsp.numCities, -1);

                // 复制交叉区间内的基因：子代1复制父代1，子代2复制父代2
                for (int i = point1; i <= point2; i++) {
                    child1Chromosome[i] = pa.chromosome[i];
                    child2Chromosome[i] = pb.chromosome[i];
                }

                // 填充子代1：按父代2的顺序填充未被复制的基因
                int currentIndex = (point2 + 1) % tsp.numCities;
                for (int i = 0; i < tsp.numCities; i++) {
                    int candidateIndex = (point2 + 1 + i) % tsp.numCities;
                    int candidate = pb.chromosome[candidateIndex];
                    // 如果 candidate 不在子代1的交叉区间内，则填入
                    if (std::find(child1Chromosome.begin(), child1Chromosome.end(), candidate) == child1Chromosome.end()) {
                        child1Chromosome[currentIndex] = candidate;
                        currentIndex = (currentIndex + 1) % tsp.numCities;
                    }
                }

                // 填充子代2：按父代1的顺序填充未被复制的基因
                currentIndex = (point2 + 1) % tsp.numCities;
                for (int i = 0; i < tsp.numCities; i++) {
                    int candidateIndex = (point2 + 1 + i) % tsp.numCities;
                    int candidate = pa.chromosome[candidateIndex];
                    if (std::find(child2Chromosome.begin(), child2Chromosome.end(), candidate) == child2Chromosome.end()) {
                        child2Chromosome[currentIndex] = candidate;
                        currentIndex = (currentIndex + 1) % tsp.numCities;
                    }
                }

                // 更新子代的染色体
                child1.chromosome = child1Chromosome;
                child2.chromosome = child2Chromosome;
            }
            offspring[island].push_back(child1);
            offspring[island].push_back(child2);
        }
    }
    return offspring;
}

// 3. Mutation: 均匀变异，每个基因有一定概率交换位置
void mutationCPU(const TSP &tsp, Offspring &offspring) {
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

// 4. Replacement: 用适应度更高的后代替换父代
void replacementCPU(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring) {
    for (int island = 0; island < tsp.numIslands; island++) {
        int numPairs = parentPairs[island].size();
        for (int i = 0; i < numPairs; i++) {
            const Individual &pa = parentPairs[island][i].first;
            const Individual &pb = parentPairs[island][i].second;
            const Individual &child1 = offspring[island][2*i];
            const Individual &child2 = offspring[island][2*i+1];
            float paFit = computeFitnessCPU(pa, tsp);
            float pbFit = computeFitnessCPU(pb, tsp);
            float c1Fit = computeFitnessCPU(child1, tsp);
            float c2Fit = computeFitnessCPU(child2, tsp);
            if (c1Fit > paFit) {
                for (auto &ind : tsp.population[island]) {
                    if (ind.chromosome == pa.chromosome) {
                        ind = child1;
                        break;
                    }
                }
            }
            if (c2Fit > pbFit) {
                for (auto &ind : tsp.population[island]) {
                    if (ind.chromosome == pb.chromosome) {
                        ind = child2;
                        break;
                    }
                }
            }
        }
    }
}

// 5. Migration: 岛间迁移（有向环拓扑结构）
void migrationCPU(TSP &tsp) {
    int numIslands = tsp.numIslands;
    std::vector<Individual> bestIndividuals(numIslands);
    std::vector<int> worstIndex(numIslands, -1);
    for (int island = 0; island < numIslands; island++) {
        float bestFit = -1.0f;
        float worstFit = 1e9;
        int worstIdx = -1;
        for (int i = 0; i < tsp.population[island].size(); i++) {
            float fit = computeFitnessCPU(tsp.population[island][i], tsp);
            if (fit > bestFit) {
                bestFit = fit;
                bestIndividuals[island] = tsp.population[island][i];
            }
            if (fit < worstFit) {
                worstFit = fit;
                worstIdx = i;
            }
        }
        worstIndex[island] = worstIdx;
    }
    for (int island = 0; island < numIslands; island++) {
        int prevIsland = (island - 1 + numIslands) % numIslands;
        float incomingFit = computeFitnessCPU(bestIndividuals[prevIsland], tsp);
        float currentWorstFit = computeFitnessCPU(tsp.population[island][worstIndex[island]], tsp);
        if (incomingFit > currentWorstFit) {
            tsp.population[island][worstIndex[island]] = bestIndividuals[prevIsland];
        }
    }
}

} // namespace GA
