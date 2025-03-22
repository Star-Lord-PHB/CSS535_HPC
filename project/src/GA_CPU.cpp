#include "GA_CPU.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include "TSP.h"

namespace GA {

    // ---------------------------------------------------------------------
    // Helper function: Compute the fitness of a single individual (CPU version)
    // Fitness is defined as 1/(total tour distance)
    // ---------------------------------------------------------------------
    static float computeFitnessCPU(const Individual &ind, const TSP &tsp) {
        float totalDistance = 0.0f;
        for (int i = 0; i < tsp.numCities - 1; i++) {
            int city1 = ind.chromosome[i];
            int city2 = ind.chromosome[i + 1];
            totalDistance += tsp.distanceMatrix[city1][city2];
        }
        // Add the distance from the last city back to the first
        totalDistance += tsp.distanceMatrix[ind.chromosome[tsp.numCities - 1]][ind.chromosome[0]];
        return (totalDistance <= 0.0f) ? 0.0f : 1.0f / totalDistance;
    }

    // ---------------------------------------------------------------------
    // 1) Selection (CPU)
    // Shuffles individuals within each island and pairs them.
    // The results are stored in tsp.parentPairs, and the pair counts are updated.
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
            // If the island has an odd number of individuals, pair the last with itself.
            if (islandPop.size() % 2 == 1) {
                tsp.parentPairs[island].push_back({ islandPop.back(), islandPop.back() });
            }
        }
        // Update parentPairCount vector
        tsp.parentPairCount.clear();
        for (int island = 0; island < tsp.numIslands; island++) {
            tsp.parentPairCount.push_back(tsp.parentPairs[island].size());
        }
    }

    // ---------------------------------------------------------------------
    // 2) Crossover (CPU)
    // Uses tsp.parentPairs to generate offspring and stores the result in tsp.offsprings.
    // Order Crossover (OX) is applied for each parent pair.
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
                Individual child1 = pa, child2 = pb; // Start as copies of parents
                if (probDist(rng) < tsp.crossoverProbability) {
                    int point1 = pointDist(rng);
                    int point2 = pointDist(rng);
                    if (point1 > point2) std::swap(point1, point2);

                    std::vector<int> ch1(tsp.numCities, -1), ch2(tsp.numCities, -1);
                    // Copy the segment from each parent
                    for (int k = point1; k <= point2; k++) {
                        ch1[k] = pa.chromosome[k];
                        ch2[k] = pb.chromosome[k];
                    }
                    // Fill child1 using genes from parent B in order
                    int index = (point2 + 1) % tsp.numCities;
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pb.chromosome[idx];
                        if (std::find(ch1.begin(), ch1.end(), gene) == ch1.end()) {
                            ch1[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    // Fill child2 using genes from parent A in order
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
                // Reset offspring fitness (to be recalculated later)
                child1.fitness = 0.0f;
                child2.fitness = 0.0f;
                child1.islandID = pa.islandID;
                child2.islandID = pa.islandID;
                tsp.offsprings[island].push_back(child1);
                tsp.offsprings[island].push_back(child2);
            }
        }
        // (Optionally, update a flattened offspring array here if needed)
    }

    // ---------------------------------------------------------------------
    // 3) Mutation (CPU)
    // Mutates the offspring in tsp.offsprings by randomly swapping genes.
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
    }

    // ---------------------------------------------------------------------
    // 4) Replacement (CPU)
    // For each parent pair and corresponding offspring in tsp.parentPairs and tsp.offsprings,
    // select the two best individuals (highest fitness) from the set {parentA, parentB, child1, child2}
    // and update tsp.population accordingly. After replacement, the population is re-flattened.
    // ---------------------------------------------------------------------
    void replacementCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < tsp.parentPairs[island].size(); i++) {
                // Create an array of 4 candidates: parent A, parent B, child1, and child2.
                Individual candidates[4] = {
                    tsp.parentPairs[island][i].first,    // parent A
                    tsp.parentPairs[island][i].second,   // parent B
                    tsp.offsprings[island][2 * i],         // child1
                    tsp.offsprings[island][2 * i + 1]       // child2
                };
                // Sort candidates in descending order based on fitness
                std::sort(candidates, candidates + 4, [](const Individual &a, const Individual &b) {
                    return a.fitness > b.fitness;
                });
                // Replace the original parents in the population with the top two candidates
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
        // Update the flattened population data stored in tsp.populationFlat
        tsp.flattenPopulationToHost();
    }

    // ---------------------------------------------------------------------
    // 5) Migration (CPU)
    // A simple ring migration example:
    // For each island, find the best individual and the worst individual.
    // Replace the worst individual of the current island with the best individual from the previous island.
    // ---------------------------------------------------------------------
    void migrationCPU(TSP &tsp) {
        int nIslands = tsp.numIslands;
        std::vector<Individual> bestInds(nIslands);
        std::vector<int> worstIndex(nIslands, -1);

        // Identify the best and worst individuals for each island.
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
        // Perform migration: Replace the worst individual of each island with the best individual from the previous island.
        for (int island = 0; island < nIslands; island++) {
            int prev = (island - 1 + nIslands) % nIslands;
            if (bestInds[prev].fitness > tsp.population[island][worstIndex[island]].fitness) {
                tsp.population[island][worstIndex[island]] = bestInds[prev];
            }
        }
        // Update the flattened population after migration.
        tsp.flattenPopulationToHost();
    }

    // ---------------------------------------------------------------------
    // Update Population Fitness (CPU)
    // Computes and updates the fitness for each individual in tsp.population,
    // then re-flattens the population data.
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
    // Update Offspring Fitness (CPU)
    // Computes and updates the fitness for each offspring in tsp.offsprings.
    // ---------------------------------------------------------------------
    void updateOffspringFitnessCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : tsp.offsprings[island]) {
                child.fitness = computeFitnessCPU(child, tsp);
            }
        }
        // Optionally, flatten offspring data if needed.
    }

} // namespace GA
