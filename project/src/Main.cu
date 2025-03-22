#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include "TSP.h"
#include "GA_CPU.h"
#include "GA_CUDA.h"
#include "GA_oneThreadPerGene.hpp"

using namespace std::chrono;

// Extended implementation strategy enum
enum class Implementation {
    CPU,
    CUDA,
    CUDA_PER_GENE,
    CUDA_SELECTION,        // Only selection using CUDA; others use CPU
    CUDA_CROSS,            // Only crossover using CUDA; others use CPU
    CUDA_MUTATION,         // Only mutation using CUDA; others use CPU
    CUDA_REPLACEMENT,      // Only replacement using CUDA; others use CPU
    CUDA_MIGRATION,        // Only migration using CUDA; others use CPU
    CUDA_UPDATE_POPULATION,  // Only update population fitness using CUDA; others use CPU
    CUDA_UPDATE_OFFSPRING,    // Only update offspring fitness using CUDA; others use CPU
    CUDA_UPDATE_ALL_FITNESS // Use CUDA for updating all fitness values
};

// Structure to hold function pointers for the GA steps
struct GAFunctionSet {
    void (*selection)(TSP &);
    void (*crossover)(TSP &);
    void (*mutation)(TSP &);
    void (*replacement)(TSP &);
    void (*migration)(TSP &);
    void (*updatePopulationFitness)(TSP &);
    void (*updateOffspringFitness)(TSP &);
};

// Extended GAResult structure to hold the result along with timing information
struct GAResult {
    std::string strategyName;
    double bestFitness;
    double totalTime; // Total time for the GA iterations (in seconds)
    double totalKernelTime;
    // Hyperparameters
    int numCities;
    int popSize;
    int mapSize;
    int numIslands;
    float parentSelectionRate;
    float crossoverProbability;
    float mutationProbability;
    int generations;
    // Timing for each phase (in seconds)
    double sel_compute, sel_kernel, sel_total;
    double cross_compute, cross_kernel, cross_total;
    double mut_compute, mut_kernel, mut_total;
    double repl_compute, repl_kernel, repl_total;
    double mig_compute, mig_kernel, mig_total;
    double updPop_compute, updPop_kernel, updPop_total;
    double updOff_compute, updOff_kernel, updOff_total;
};

// Get the strategy name string from the enum
std::string getStrategyName(Implementation impl) {
    switch (impl) {
        case Implementation::CPU: return "CPU";
        case Implementation::CUDA: return "CUDA";
        case Implementation::CUDA_SELECTION: return "CUDA_SELECTION";
        case Implementation::CUDA_CROSS: return "CUDA_CROSS";
        case Implementation::CUDA_MUTATION: return "CUDA_MUTATION";
        case Implementation::CUDA_REPLACEMENT: return "CUDA_REPLACEMENT";
        case Implementation::CUDA_MIGRATION: return "CUDA_MIGRATION";
        case Implementation::CUDA_UPDATE_POPULATION: return "CUDA_UPDATE_POPULATION";
        case Implementation::CUDA_UPDATE_OFFSPRING: return "CUDA_UPDATE_OFFSPRING";
        case Implementation::CUDA_UPDATE_ALL_FITNESS: return "CUDA_UPDATE_ALL_FITNESS";
        default: return "Unknown";
    }
}


GAResult runOneGenePerThread(TSP& tsp, const int generations) {

    auto const solution = new Array3D<int>(tsp.numIslands, tsp.popSize / tsp.numIslands, tsp.numCities + 1);
    auto const distanceMat = new Array2D<float>(tsp.numCities, tsp.numCities);
    distanceMat->fill(tsp.distanceMatrixFlat);

    auto calculationTime = milliseconds();
    auto totalTime = milliseconds();

    ga_one_thread_per_gene(
        *solution,
        *distanceMat,
        generations, tsp.crossoverProbability, tsp.mutationProbability,
        &calculationTime,
        &totalTime
    );

    unsigned bestIslandId = 0;
    unsigned bestIndividualId = 0;
    double bestFitness = INFINITY;
    for (unsigned islandId = 0; islandId < solution->xSize(); islandId++) {
        for (unsigned individualId = 0; individualId < solution->ySize(); individualId++) {
            const auto fitness = *reinterpret_cast<float*>(&solution->at(islandId, individualId, tsp.numCities));
            if (fitness < bestFitness) {
                bestIslandId = islandId;
                bestIndividualId = individualId;
                bestFitness = fitness;
            }
        }
    }

    const auto bestIndividual = std::vector(
        &solution->at(bestIslandId, bestIndividualId, 0),
        &solution->at(bestIslandId, bestIndividualId, tsp.numCities - 1)
    );



    auto result = GAResult();
    result.strategyName = "ONE_GENE_PRE_THREAD";
    result.totalTime = static_cast<double>(totalTime.count()) / 1000;
    result.bestFitness = 1 / bestFitness;
    result.totalKernelTime = static_cast<double>(calculationTime.count()) / 1000;

    delete solution;

    return result;

}


// Run a single strategy; returns a GAResult containing the result and timing info.
GAResult run_strategy(Implementation impl,
                      int numCities,
                      int popSize,
                      int mapSize,
                      int numIslands,
                      float parentSelectionRate,
                      float crossoverProbability,
                      float mutationProbability,
                      int generations)
{
    GAFunctionSet gaFuncs;
    // Set function pointers according to the selected strategy
    switch (impl) {
        case Implementation::CPU:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA:
            gaFuncs.selection = GA::selectionCUDA;
            gaFuncs.crossover = GA::crossoverCUDA;
            gaFuncs.mutation = GA::mutationCUDA;
            gaFuncs.replacement = GA::replacementCUDA;
            gaFuncs.migration = GA::migrationCUDA;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
            break;
        case Implementation::CUDA_SELECTION:
            gaFuncs.selection = GA::selectionCUDA;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_CROSS:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCUDA;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_MUTATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCUDA;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_REPLACEMENT:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCUDA;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_MIGRATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCUDA;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_UPDATE_POPULATION:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
        case Implementation::CUDA_UPDATE_OFFSPRING:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
            break;
        case Implementation::CUDA_UPDATE_ALL_FITNESS:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCUDA;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCUDA;
            break;
        default:
            gaFuncs.selection = GA::selectionCPU;
            gaFuncs.crossover = GA::crossoverCPU;
            gaFuncs.mutation = GA::mutationCPU;
            gaFuncs.replacement = GA::replacementCPU;
            gaFuncs.migration = GA::migrationCPU;
            gaFuncs.updatePopulationFitness = GA::updatePopulationFitnessCPU;
            gaFuncs.updateOffspringFitness = GA::updateOffspringFitnessCPU;
            break;
    }

    // Create a TSP instance (constructor initializes cities, population, distance matrix, GPU buffers, and time records)
    TSP tsp(numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability);

    // One Gene Per Thread (CUDA) implementation
    if (impl == Implementation::CUDA_PER_GENE) {
        return runOneGenePerThread(tsp, generations);
    }

    // Initial population fitness update
    gaFuncs.updatePopulationFitness(tsp);

    // Record the start time for the GA iterations (this covers the whole GA loop)
    auto startTime = high_resolution_clock::now();

    // GA main loop: each generation runs all GA steps
    for (int gen = 0; gen < generations; gen++) {
        gaFuncs.selection(tsp);
        gaFuncs.crossover(tsp);
        gaFuncs.mutation(tsp);
        gaFuncs.updateOffspringFitness(tsp);
        gaFuncs.replacement(tsp);
        gaFuncs.migration(tsp);
        gaFuncs.updatePopulationFitness(tsp);
    }

    // Record the end time
    auto endTime = high_resolution_clock::now();
    duration<double> duration = endTime - startTime;

    // Find the best fitness among all islands
    double bestFitness = -1.0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (const auto &ind : tsp.population[island]) {
            if (ind.fitness > bestFitness)
                bestFitness = ind.fitness;
        }
    }

    // Construct GAResult including timing data from each phase
    GAResult result;
    result.strategyName = getStrategyName(impl);
    result.bestFitness = bestFitness;
    result.totalTime = duration.count();
    result.numCities = numCities;
    result.popSize = popSize;
    result.mapSize = mapSize;
    result.numIslands = numIslands;
    result.parentSelectionRate = parentSelectionRate;
    result.crossoverProbability = crossoverProbability;
    result.mutationProbability = mutationProbability;
    result.generations = generations;

    result.sel_compute = tsp.selectionTime.computeTime;
    result.sel_kernel  = tsp.selectionTime.kernelTime;
    result.sel_total   = tsp.selectionTime.totalTime;

    result.cross_compute = tsp.crossoverTime.computeTime;
    result.cross_kernel  = tsp.crossoverTime.kernelTime;
    result.cross_total   = tsp.crossoverTime.totalTime;

    result.mut_compute = tsp.mutationTime.computeTime;
    result.mut_kernel  = tsp.mutationTime.kernelTime;
    result.mut_total   = tsp.mutationTime.totalTime;

    result.repl_compute = tsp.replacementTime.computeTime;
    result.repl_kernel  = tsp.replacementTime.kernelTime;
    result.repl_total   = tsp.replacementTime.totalTime;

    result.mig_compute = tsp.migrationTime.computeTime;
    result.mig_kernel  = tsp.migrationTime.kernelTime;
    result.mig_total   = tsp.migrationTime.totalTime;

    result.updPop_compute = tsp.updatePopulationFitnessTime.computeTime;
    result.updPop_kernel  = tsp.updatePopulationFitnessTime.kernelTime;
    result.updPop_total   = tsp.updatePopulationFitnessTime.totalTime;

    result.updOff_compute = tsp.updateOffspringFitnessTime.computeTime;
    result.updOff_kernel  = tsp.updateOffspringFitnessTime.kernelTime;
    result.updOff_total   = tsp.updateOffspringFitnessTime.totalTime;

    return result;
}


int main() {
    // Hyperparameters
    int numCities = 256;
    int popSize = 8192;
    int mapSize = 1000;
    int numIslands = 16;
    float parentSelectionRate = 1.0f;
    float crossoverProbability = 0.7f;
    float mutationProbability = 0.05f;
    int generations = 500;

    constexpr bool runOneGenePerThreadImplementation = true;

    if (runOneGenePerThreadImplementation) {

        const auto result = run_strategy(
            Implementation::CUDA_PER_GENE,
            numCities, popSize, mapSize, numIslands,
            parentSelectionRate, crossoverProbability, mutationProbability, generations
        );

        std::cout << "One Gene Per Thread (CUDA) Result:" << std::endl;
        std::cout << "Time Total: " << result.totalTime << " seconds" << std::endl;
        std::cout << "Time Total Kernel: " << result.totalKernelTime << " seconds" << std::endl;
        std::cout << "Best Fitness: " << result.bestFitness << std::endl;

        return 0;

    }

    // Strategy list
    std::vector strategies = {
        Implementation::CPU,
        Implementation::CUDA,
        Implementation::CUDA_SELECTION,
        Implementation::CUDA_CROSS,
        Implementation::CUDA_MUTATION,
        Implementation::CUDA_REPLACEMENT,
        Implementation::CUDA_MIGRATION,
        Implementation::CUDA_UPDATE_POPULATION,
        Implementation::CUDA_UPDATE_OFFSPRING,
        Implementation::CUDA_UPDATE_ALL_FITNESS
    };

    // Open CSV file to write data
    std::ofstream outfile("D://CSS535//CSS535_HPC//project//data.csv");
    outfile << "strategy,best_fitness,total_time,numCities,popSize,mapSize,numIslands,parentSelectionRate,crossoverProbability,mutationProbability,generations,"
            << "sel_compute,sel_kernel,sel_total,"
            << "cross_compute,cross_kernel,cross_total,"
            << "mut_compute,mut_kernel,mut_total,"
            << "repl_compute,repl_kernel,repl_total,"
            << "mig_compute,mig_kernel,mig_total,"
            << "updPop_compute,updPop_kernel,updPop_total,"
            << "updOff_compute,updOff_kernel,updOff_total\n";

    // Iterate through each strategy, run GA, and write results
    for (auto impl : strategies) {
        GAResult res = run_strategy(impl, numCities, popSize, mapSize, numIslands,
                                    parentSelectionRate, crossoverProbability, mutationProbability, generations);
        outfile << res.strategyName << ","
                << res.bestFitness << ","
                << res.totalTime << ","
                << res.numCities << ","
                << res.popSize << ","
                << res.mapSize << ","
                << res.numIslands << ","
                << res.parentSelectionRate << ","
                << res.crossoverProbability << ","
                << res.mutationProbability << ","
                << res.generations << ","
                << res.sel_compute << "," << res.sel_kernel << "," << res.sel_total << ","
                << res.cross_compute << "," << res.cross_kernel << "," << res.cross_total << ","
                << res.mut_compute << "," << res.mut_kernel << "," << res.mut_total << ","
                << res.repl_compute << "," << res.repl_kernel << "," << res.repl_total << ","
                << res.mig_compute << "," << res.mig_kernel << "," << res.mig_total << ","
                << res.updPop_compute << "," << res.updPop_kernel << "," << res.updPop_total << ","
                << res.updOff_compute << "," << res.updOff_kernel << "," << res.updOff_total
                << "\n";
        std::cout << "Finished strategy " << res.strategyName
                  << ": best fitness = " << res.bestFitness
                  << ", total time = " << res.totalTime << " seconds."
                  << std::endl;
    }

    outfile.close();
}
