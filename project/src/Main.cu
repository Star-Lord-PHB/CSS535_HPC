#include <iostream>
#include <chrono>
#include "TSP.h"
#include "GAInterface.h"
#include "GA_CPU.h"
#include "GA_CUDA.h"
#include "GA_oneThreadPerGene.hpp" // Additional GA variant (if needed)

// Enum to choose implementation version
enum class Implementation { CPU, CUDA };

// Function set structure to hold GA function pointers
struct GAFunctionSet {
    ParentPairs (*selection)(TSP &);
    Offspring (*crossover)(TSP &, const ParentPairs &);
    void (*mutation)(TSP &, Offspring &);
    void (*replacement)(TSP &, const ParentPairs &, const Offspring &);
    void (*migration)(TSP &);
    void (*updatePopulationFitness)(TSP &);
    void (*updateOffspringFitness)(TSP &, Offspring &);
};

int main() {
    // GA algorithm parameters
    int numCities = 256;              // Number of cities
    int popSize = 1024;               // Total population size (across all islands)
    int mapSize = 1000;               // Size of the map (e.g., cities will have coordinates in range 0 to mapSize)
    int numIslands = 64;              // Number of islands (subpopulations)
    float parentSelectionRate = 0.5f; // Parent selection rate (optional parameter)
    float crossoverProbability = 0.7f;// Crossover probability
    float mutationProbability = 0.05f; // Mutation probability
    int generations = 500;            // Number of generations to run

    // Choose implementation version: CUDA or CPU
    Implementation impl = Implementation::CUDA; // Change to Implementation::CPU to use CPU version

    // Set function pointers based on the chosen implementation
    GAFunctionSet gaFuncs;
    if (impl == Implementation::CPU) {
        gaFuncs.selection = GA::selectionCPU;
        gaFuncs.crossover = GA::crossoverCPU;
        gaFuncs.mutation = GA::mutationCPU;
        gaFuncs.replacement = GA::replacementCPU;
        gaFuncs.migration = GA::migrationCPU;
        gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
        gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
    } else { // Implementation::CUDA
        gaFuncs.selection = GA::selectionCUDA;
        gaFuncs.crossover = GA::crossoverCUDA;
        gaFuncs.mutation = GA::mutationCUDA;
        gaFuncs.replacement = GA::replacementCUDA;
        gaFuncs.migration = GA::migrationCUDA;
        gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
        gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
    }

    // Initialize TSP problem instance (constructor generates cities, population, and distance matrix)
    TSP tsp(numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability);

    // Initial fitness update: compute fitness for the entire population
    gaFuncs.updatePopulationFitness(tsp);

    // Record start time for GA iterations
    auto startTime = std::chrono::high_resolution_clock::now();

    // Main GA loop over generations
    for (int gen = 0; gen < generations; gen++) {
        // Selection: pair parents within each island
        auto parentPairs = gaFuncs.selection(tsp);
        // Crossover: generate offspring from parent pairs
        auto offspring = gaFuncs.crossover(tsp, parentPairs);
        // Mutation: apply mutation to the offspring
        gaFuncs.mutation(tsp, offspring);

        // Update fitness for the newly generated offspring only
        gaFuncs.updateOffspringFitness(tsp, offspring);

        // Replacement: replace poorer parents with the better individuals chosen from parents and offspring
        gaFuncs.replacement(tsp, parentPairs, offspring);

        // Migration: exchange individuals among islands (if applicable)
        gaFuncs.migration(tsp);

        // Update the fitness for the entire population after replacement and migration
        gaFuncs.updatePopulationFitness(tsp);

        std::cout << "Generation " << gen << " complete." << std::endl;
        // Optionally, print the best fitness on each island
        // for (int island = 0; island < tsp.numIslands; island++) {
        //     float bestFit = -1.0f;
        //     for (const auto &ind : tsp.population[island]) {
        //         if (ind.fitness > bestFit)
        //             bestFit = ind.fitness;
        //     }
        //     std::cout << "  Island " << island << " best fitness: " << bestFit << std::endl;
        // }
    }

    // Record end time for GA iterations
    auto endTime = std::chrono::high_resolution_clock::now();
    // Compute the total duration of GA iterations in seconds
    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "Total GA iterations time: " << duration.count() << " seconds." << std::endl;

    return 0;
}
