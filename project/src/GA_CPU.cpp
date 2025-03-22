#include "GA_CPU.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include "TSP.h"

namespace GA {

    // ------------------------------
    // Helper function: Compute the fitness of a single individual (CPU version)
    // Fitness is defined as 1 / (total distance of the tour)
    // ------------------------------
    static float computeFitnessCPU(const Individual &ind, const TSP &tsp) {
        float totalDistance = 0.0f;
        for (int i = 0; i < tsp.numCities - 1; i++) {
            int city1 = ind.chromosome[i];
            int city2 = ind.chromosome[i + 1];
            totalDistance += tsp.distanceMatrix[city1][city2];
        }
        // Add distance from the last city back to the first city
        totalDistance += tsp.distanceMatrix[ind.chromosome[tsp.numCities - 1]][ind.chromosome[0]];
        return (totalDistance <= 0.0f) ? 0.0f : 1.0f / totalDistance;
    }

    // ------------------------------
    // 1) Selection (CPU)
    // Randomly shuffles the individuals in each island and pairs them.
    // If there is an odd number, the last individual is paired with itself.
    // ------------------------------
    ParentPairs selectionCPU(TSP &tsp) {
        ParentPairs parentPairs(tsp.numIslands);
        std::mt19937 rng(std::random_device{}());

        for (int island = 0; island < tsp.numIslands; island++) {
            auto &islandPop = tsp.population[island];
            std::shuffle(islandPop.begin(), islandPop.end(), rng);
            int numPairs = islandPop.size() / 2;
            for (int i = 0; i < numPairs; i++) {
                parentPairs[island].push_back({ islandPop[2 * i], islandPop[2 * i + 1] });
            }
            // If the island has an odd number of individuals, pair the last individual with itself.
            if (islandPop.size() % 2 == 1) {
                parentPairs[island].push_back({ islandPop.back(), islandPop.back() });
            }
        }
        return parentPairs;
    }

    // ------------------------------
    // 2) Crossover (CPU)
    // Performs Order Crossover (OX) for each parent pair.
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
                    // Order Crossover (OX)
                    int point1 = pointDist(rng);
                    int point2 = pointDist(rng);
                    if (point1 > point2) std::swap(point1, point2);

                    std::vector<int> ch1(tsp.numCities, -1), ch2(tsp.numCities, -1);
                    // Copy the segment from parent A and parent B respectively
                    for (int k = point1; k <= point2; k++) {
                        ch1[k] = pa.chromosome[k];
                        ch2[k] = pb.chromosome[k];
                    }
                    // Fill child1: take genes from parent B in order
                    int index = (point2 + 1) % tsp.numCities;
                    for (int k = 0; k < tsp.numCities; k++) {
                        int idx = (point2 + 1 + k) % tsp.numCities;
                        int gene = pb.chromosome[idx];
                        if (std::find(ch1.begin(), ch1.end(), gene) == ch1.end()) {
                            ch1[index] = gene;
                            index = (index + 1) % tsp.numCities;
                        }
                    }
                    // Fill child2: take genes from parent A in order
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
                // Reset offspring fitness to 0 (to be recalculated later)
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
    // Randomly swaps genes within each offspring with a certain mutation probability.
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
    // For each parent pair, consider the two parents and their two offspring.
    // Select the two best individuals (highest fitness) to replace the original parents.
    // ------------------------------
    void replacementCPU(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < parentPairs[island].size(); i++) {
                // Construct an array of four candidates: parent A, parent B, child1, and child2.
                Individual candidates[4] = {
                    parentPairs[island][i].first,   // parent A
                    parentPairs[island][i].second,  // parent B
                    offspring[island][2 * i],         // child1
                    offspring[island][2 * i + 1]      // child2
                };
                // Sort candidates in descending order based on fitness
                std::sort(candidates, candidates + 4,
                          [](const Individual &a, const Individual &b) {
                              return a.fitness > b.fitness; // higher fitness first
                          });

                // Replace the original parents in the population with the top two candidates
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
        // Update the flattened population data after replacement
        tsp.flattenPopulationToHost();
    }

    // ------------------------------
    // 5) Migration (CPU)
    // A simple ring migration example: for each island, replace the worst individual with the best individual from the previous island.
    // ------------------------------
    void migrationCPU(TSP &tsp) {
        int nIslands = tsp.numIslands;
        std::vector<Individual> bestInds(nIslands);
        std::vector<int> worstIndex(nIslands, -1);

        // For each island, find the best individual and record the index of the worst individual.
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
        // Migration: Replace the worst individual of the current island with the best individual from the previous island.
        for (int island = 0; island < nIslands; island++) {
            int prev = (island - 1 + nIslands) % nIslands;
            if (bestInds[prev].fitness > tsp.population[island][worstIndex[island]].fitness) {
                tsp.population[island][worstIndex[island]] = bestInds[prev];
            }
        }
        // Synchronize the flattened population after migration
        tsp.flattenPopulationToHost();
    }

    // ------------------------------
    // Update Population Fitness (CPU)
    // Computes the fitness for each individual in the population.
    // ------------------------------
    void updatePopulationFitnessCPU(TSP &tsp) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                ind.fitness = computeFitnessCPU(ind, tsp);
            }
        }
        // After computing fitness, you may update the flattened population (populationFlat) if needed.
    }

    // ------------------------------
    // Update Offspring Fitness (CPU)
    // Computes the fitness for each offspring.
    // ------------------------------
    void updateOffspringFitnessCPU(TSP &tsp, Offspring &offspring) {
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                child.fitness = computeFitnessCPU(child, tsp);
            }
        }
    }

} // namespace GA
