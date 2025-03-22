#include "GA_CUDA.h"
#include "GA_CPU.h"  // Refer to the CPU version for synchronization functions
#include "TSP.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>
#include <iostream>
#include <chrono>

namespace GA {

using namespace std::chrono;

// ---------------------------------------------------------------------
// Kernel: Compute fitness for each individual
// ---------------------------------------------------------------------
__global__ void computeFitnessKernel(const int *d_population,
                                       const float *d_distanceMatrix,
                                       float *d_fitness,
                                       int numCities,
                                       int popCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popCount) {
        float totalDist = 0.0f;
        int base = idx * numCities;
        for (int i = 0; i < numCities - 1; i++) {
            int c1 = d_population[base + i];
            int c2 = d_population[base + i + 1];
            totalDist += d_distanceMatrix[c1 * numCities + c2];
        }
        int lastCity = d_population[base + numCities - 1];
        int firstCity = d_population[base];
        totalDist += d_distanceMatrix[lastCity * numCities + firstCity];
        d_fitness[idx] = (totalDist <= 0.0f) ? 0.0f : (1.0f / totalDist);
    }
}

// ---------------------------------------------------------------------
// Kernel: Order Crossover (OX) for a pair of parents.
// ---------------------------------------------------------------------
__global__ void crossoverKernel(const int *d_parentA, const int *d_parentB,
                                  int *d_child1, int *d_child2,
                                  int numPairs, int numCities,
                                  float crossoverProb, unsigned long seed)
{
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    // Initialize CURAND state with given seed and thread-specific offset
    curandState state;
    curand_init(seed, pairIdx, 0, &state);

    int base = pairIdx * numCities;
    int *child1 = d_child1 + base;
    int *child2 = d_child2 + base;

    float r = curand_uniform(&state);
    // If crossover is not applied, copy parents directly to children
    if (r >= crossoverProb) {
        for (int i = 0; i < numCities; i++) {
            child1[i] = d_parentA[base + i];
            child2[i] = d_parentB[base + i];
        }
        return;
    }

    // Check if parents are identical; if yes, copy them directly
    bool identical = true;
    for (int i = 0; i < numCities; i++) {
        if (d_parentA[base + i] != d_parentB[base + i]) {
            identical = false;
            break;
        }
    }
    if (identical) {
        for (int i = 0; i < numCities; i++) {
            child1[i] = d_parentA[base + i];
            child2[i] = d_parentB[base + i];
        }
        return;
    }

    // Initialize children with -1 values
    for (int i = 0; i < numCities; i++) {
        child1[i] = -1;
        child2[i] = -1;
    }
    int p1 = curand(&state) % numCities;
    int p2 = curand(&state) % numCities;
    if (p1 > p2) { int tmp = p1; p1 = p2; p2 = tmp; }
    // Copy segment from parents to children
    for (int i = p1; i <= p2; i++) {
        child1[i] = d_parentA[base + i];
        child2[i] = d_parentB[base + i];
    }
    // Fill child1 with genes from parentB
    int idx = (p2 + 1) % numCities;
    for (int i = 0; i < numCities; i++) {
        int pos = (p2 + 1 + i) % numCities;
        int gene = d_parentB[base + pos];
        bool found = false;
        for (int j = p1; j <= p2; j++) {
            if (child1[j] == gene) { found = true; break; }
        }
        if (!found) {
            child1[idx] = gene;
            idx = (idx + 1) % numCities;
        }
    }
    // Fill child2 with genes from parentA
    idx = (p2 + 1) % numCities;
    for (int i = 0; i < numCities; i++) {
        int pos = (p2 + 1 + i) % numCities;
        int gene = d_parentA[base + pos];
        bool found = false;
        for (int j = p1; j <= p2; j++) {
            if (child2[j] == gene) { found = true; break; }
        }
        if (!found) {
            child2[idx] = gene;
            idx = (idx + 1) % numCities;
        }
    }
}

// ---------------------------------------------------------------------
// Kernel: Mutation: Each thread processes one individual
// ---------------------------------------------------------------------
__global__ void mutationKernel(int *d_offspring, int totalIndividuals, int numCities,
                               float mutationProb, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalIndividuals) return;

    // Initialize CURAND state for this thread
    curandState state;
    curand_init(seed, idx, 0, &state);

    int start = idx * numCities;
    for (int i = 0; i < numCities; i++) {
        float r = curand_uniform(&state);
        if (r < mutationProb) {
            int j = curand(&state) % numCities;
            int tmp = d_offspring[start + i];
            d_offspring[start + i] = d_offspring[start + j];
            d_offspring[start + j] = tmp;
        }
    }
}

// ---------------------------------------------------------------------
// Kernel: Replacement
// Each thread processes one parent pair (with two children).
// It selects the two best individuals from {parent A, parent B, child1, child2}
// and writes them back to the flattened population.
// ---------------------------------------------------------------------
__global__ void replacementKernel(
    int *d_population, float *d_populationFitness,
    const int *d_parentA, const int *d_parentB,
    const float *d_parentFitness,
    const int *d_child1, const int *d_child2,
    const float *d_childFitness,
    int numPairs, int numCities, int totalPairs)
{
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    int base = pairIdx * numCities;
    float fits[4] = {
        d_parentFitness[2 * pairIdx],
        d_parentFitness[2 * pairIdx + 1],
        d_childFitness[pairIdx],
        d_childFitness[totalPairs + pairIdx]
    };
    const int* chrom[4] = {
        d_parentA + base,
        d_parentB + base,
        d_child1 + base,
        d_child2 + base
    };

    int bestIdx = 0, secondIdx = 1;
    if (fits[secondIdx] > fits[bestIdx]) {
        int tmp = bestIdx; bestIdx = secondIdx; secondIdx = tmp;
    }
    for (int i = 2; i < 4; i++) {
        if (fits[i] > fits[bestIdx]) {
            secondIdx = bestIdx;
            bestIdx = i;
        } else if (fits[i] > fits[secondIdx]) {
            secondIdx = i;
        }
    }
    int popOffsetA = pairIdx * 2 * numCities;
    int popOffsetB = popOffsetA + numCities;
    for (int i = 0; i < numCities; i++) {
        d_population[popOffsetA + i] = chrom[bestIdx][i];
        d_population[popOffsetB + i] = chrom[secondIdx][i];
    }
    d_populationFitness[2 * pairIdx] = fits[bestIdx];
    d_populationFitness[2 * pairIdx + 1] = fits[secondIdx];
}

// ====================== CUDA Version Functions with Timing ======================
//
// For the CUDA version, computeTime records the time from the beginning of the function
// until the GPU results are mapped back to the CPU (including CPU->GPU transfers,
// kernel launch and execution). The kernelTime is measured using CUDA events.
// The totalTime is the total time from the start of the function to the end,
// including the mapping of GPU data back to the CPU.

// selectionCUDA: Calls the CPU version of selection (pure CPU phase, kernelTime = 0)
void selectionCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    auto compute_start = high_resolution_clock::now();
    selectionCPU(tsp);  // Calls CPU version
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - compute_start).count();
    // No GPU data mapping needed here, so totalTime equals compTime.
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    tsp.selectionTime.computeTime += compTime;
    tsp.selectionTime.kernelTime += 0;
    tsp.selectionTime.totalTime += totTime;
}

// crossoverCUDA: Includes CPU->GPU transfer, kernel execution, and mapping results back to CPU.
// The CPU mapping portion is not included in computeTime.
void crossoverCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    auto compute_start = high_resolution_clock::now();
    // 1. Transfer latest flattened parent data from CPU to GPU (included in computeTime)
    cudaMemcpy(tsp.d_parentA, tsp.parentAFlat.data(), tsp.parentAFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tsp.d_parentB, tsp.parentBFlat.data(), tsp.parentBFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
    int totalPairs = 0;
    for (int i = 0; i < tsp.numIslands; i++) {
        totalPairs += tsp.parentPairs[i].size();
    }
    if (totalPairs == 0) {
        std::cerr << "Warning: No parent pairs available for crossover." << std::endl;
        tsp.offsprings.clear();
        tsp.offsprings.resize(tsp.numIslands);
        auto compute_end = high_resolution_clock::now();
        double compTime = duration_cast<duration<double>>(compute_end - compute_start).count();
        auto total_end = high_resolution_clock::now();
        double totTime = duration_cast<duration<double>>(total_end - total_start).count();
        tsp.crossoverTime.computeTime += compTime;
        tsp.crossoverTime.kernelTime += 0;
        tsp.crossoverTime.totalTime += totTime;
        return;
    }
    int threads = 256;
    int blocks = (totalPairs + threads - 1) / threads;
    unsigned long seed = time(nullptr);

    // Measure kernel execution time using CUDA events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    crossoverKernel<<<blocks, threads>>>(tsp.d_parentA, tsp.d_parentB, tsp.d_child1, tsp.d_child2,
                                           totalPairs, tsp.numCities, tsp.crossoverProbability, seed);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelElapsed = 0;
    cudaEventElapsedTime(&kernelElapsed, startEvent, stopEvent); // in milliseconds
    double kTime = kernelElapsed / 1000.0;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaDeviceSynchronize();
    // 2. Transfer results from GPU to CPU (this mapping is not included in computeTime)
    tsp.offspringFlat.resize(totalPairs * 2 * tsp.numCities);
    cudaMemcpy(tsp.offspringFlat.data(), tsp.d_child1, totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsp.offspringFlat.data() + totalPairs * tsp.numCities, tsp.d_child2,
               totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - compute_start).count();
    // 3. CPU mapping phase: reconstruct offspring structure (not included in computeTime)
    auto mapping_start = high_resolution_clock::now();
    Offspring offsprings;
    offsprings.resize(tsp.numIslands);
    int pairIndex = 0;
    for (int island = 0; island < tsp.numIslands; island++) {
        int numPairs = tsp.parentPairs[island].size();
        for (int i = 0; i < numPairs; i++) {
            Individual child1, child2;
            child1.chromosome.resize(tsp.numCities);
            child2.chromosome.resize(tsp.numCities);
            for (int j = 0; j < tsp.numCities; j++) {
                child1.chromosome[j] = tsp.offspringFlat[pairIndex * tsp.numCities + j];
                child2.chromosome[j] = tsp.offspringFlat[(totalPairs + pairIndex) * tsp.numCities + j];
            }
            child1.fitness = 0.0f;
            child2.fitness = 0.0f;
            child1.islandID = island;
            child2.islandID = island;
            offsprings[island].push_back(child1);
            offsprings[island].push_back(child2);
            pairIndex++;
        }
    }
    auto mapping_end = high_resolution_clock::now();
    double mappingTime = duration_cast<duration<double>>(mapping_end - mapping_start).count();
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    // computeTime includes CPU->GPU, kernel execution, and GPU->CPU transfer (but not the mapping phase)
    tsp.crossoverTime.computeTime += compTime;
    tsp.crossoverTime.kernelTime += kTime;
    tsp.crossoverTime.totalTime += totTime;
    tsp.offsprings = offsprings;
}

// mutationCUDA: Includes kernel execution and CPU update of offspring structure.
// The CPU mapping phase is not included in computeTime.
void mutationCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    auto compute_start = high_resolution_clock::now();
    int totalPairs = 0;
    for (int i = 0; i < tsp.numIslands; i++) {
        totalPairs += tsp.parentPairs[i].size();
    }
    int totalOffspring = totalPairs * 2;
    int totalGenes = tsp.offspringFlat.size(); // = totalOffspring * tsp.numCities

    // Transfer offspring data from CPU to GPU (included in computeTime)
    cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalOffspring + threads - 1) / threads;
    unsigned long seed = time(nullptr);

    // Record kernel execution time
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    mutationKernel<<<blocks, threads>>>(tsp.d_offspring, totalOffspring, tsp.numCities, tsp.mutationProbability, seed);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelElapsed = 0;
    cudaEventElapsedTime(&kernelElapsed, startEvent, stopEvent);
    double kTime = kernelElapsed / 1000.0;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaDeviceSynchronize();

    // Transfer updated offspring data from GPU to CPU (included in computeTime)
    cudaMemcpy(tsp.offspringFlat.data(), tsp.d_offspring, totalGenes * sizeof(int), cudaMemcpyDeviceToHost);
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - compute_start).count();

    // CPU mapping phase: update offspring structure (not included in computeTime)
    auto mapping_start = high_resolution_clock::now();
    int offset = 0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (auto &child : tsp.offsprings[island]) {
            for (int j = 0; j < tsp.numCities; j++) {
                child.chromosome[j] = tsp.offspringFlat[offset * tsp.numCities + j];
            }
            offset++;
        }
    }
    auto mapping_end = high_resolution_clock::now();
    double mappingTime = duration_cast<duration<double>>(mapping_end - mapping_start).count();
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    tsp.mutationTime.computeTime += compTime;
    tsp.mutationTime.kernelTime += kTime;
    tsp.mutationTime.totalTime += totTime;
}

// updateOffspringFitnessCUDA: Includes kernel execution and CPU update of offspring fitness.
// The CPU mapping phase (updating the offspring structure) is not included in computeTime.
void updateOffspringFitnessCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    auto compute_start = high_resolution_clock::now();
    int totalPairs = 0;
    for (int i = 0; i < tsp.numIslands; i++) {
        totalPairs += tsp.parentPairs[i].size();
    }
    int totalOffspring = totalPairs * 2;
    int totalGenes = tsp.offspringFlat.size();

    // Transfer offspring data from CPU to GPU (included in computeTime)
    cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalOffspring + threads - 1) / threads;

    // Record kernel execution time
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    computeFitnessKernel<<<blocks, threads>>>(tsp.d_offspring, tsp.d_distanceMatrix, tsp.d_offspringFitness, tsp.numCities, totalOffspring);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelElapsed = 0;
    cudaEventElapsedTime(&kernelElapsed, startEvent, stopEvent);
    double kTime = kernelElapsed / 1000.0;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaDeviceSynchronize();

    // Transfer computed fitness from GPU to CPU (included in computeTime)
    tsp.offspringFitnessFlat.resize(totalOffspring);
    cudaMemcpy(tsp.offspringFitnessFlat.data(), tsp.d_offspringFitness, totalOffspring * sizeof(float), cudaMemcpyDeviceToHost);
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - compute_start).count();

    // CPU mapping phase: update the offspring structure (not included in computeTime)
    auto mapping_start = high_resolution_clock::now();
    int idx = 0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (auto &child : tsp.offsprings[island]) {
            child.fitness = tsp.offspringFitnessFlat[idx++];
        }
    }
    auto mapping_end = high_resolution_clock::now();
    double mappingTime = duration_cast<duration<double>>(mapping_end - mapping_start).count();
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    tsp.updateOffspringFitnessTime.computeTime += compTime;
    tsp.updateOffspringFitnessTime.kernelTime += kTime;
    tsp.updateOffspringFitnessTime.totalTime += totTime;
}

// updatePopulationFitnessCUDA: Includes kernel execution and CPU update of population fitness.
// The CPU mapping phase (updating the population structure) is not included in computeTime.
void updatePopulationFitnessCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    // Transfer population data from CPU to GPU
    cudaMemcpy(tsp.d_population, tsp.populationFlat.data(), tsp.populationFlat.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (tsp.popSize + threads - 1) / threads;

    // Record kernel execution time
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    computeFitnessKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_distanceMatrix, tsp.d_populationFitness, tsp.numCities, tsp.popSize);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelElapsed = 0;
    cudaEventElapsedTime(&kernelElapsed, startEvent, stopEvent);
    double kTime = kernelElapsed / 1000.0;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaDeviceSynchronize();

    std::vector<float> h_fit(tsp.popSize);
    cudaMemcpy(h_fit.data(), tsp.d_populationFitness, tsp.popSize * sizeof(float), cudaMemcpyDeviceToHost);
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - total_start).count(); // Includes CPU->GPU, kernel, and GPU->CPU transfers
    // CPU mapping phase: update population structure (not included in computeTime)
    auto mapping_start = high_resolution_clock::now();
    int idx = 0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (auto &ind : tsp.population[island]) {
            ind.fitness = h_fit[idx++];
        }
    }
    auto mapping_end = high_resolution_clock::now();
    double mappingTime = duration_cast<duration<double>>(mapping_end - mapping_start).count();
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    tsp.updatePopulationFitnessTime.computeTime += compTime; // Not subtracting mapping time
    tsp.updatePopulationFitnessTime.kernelTime += kTime;
    tsp.updatePopulationFitnessTime.totalTime += totTime;
}

// replacementCUDA: Includes kernel execution and CPU update of population structure.
// The CPU mapping phase (updating the population structure) is not included in computeTime.
void replacementCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    int totalPairs = 0;
    for (int i = 0; i < tsp.numIslands; i++) {
        totalPairs += tsp.parentPairs[i].size();
    }
    // Transfer data from CPU to GPU
    cudaMemcpy(tsp.d_parentA, tsp.parentAFlat.data(), tsp.parentAFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tsp.d_parentB, tsp.parentBFlat.data(), tsp.parentBFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tsp.d_parentFitness, tsp.parentFitnessFlat.data(), tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
    int totalGenesPerChild = totalPairs * tsp.numCities;
    cudaMemcpy(tsp.d_child1, tsp.offspringFlat.data(), totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tsp.d_child2, tsp.offspringFlat.data() + totalGenesPerChild, totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tsp.d_offspringFitness, tsp.offspringFitnessFlat.data(), tsp.offspringFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalPairs + threads - 1) / threads;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    replacementKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_populationFitness,
                                             tsp.d_parentA, tsp.d_parentB, tsp.d_parentFitness,
                                             tsp.d_child1, tsp.d_child2, tsp.d_offspringFitness,
                                             totalPairs, tsp.numCities, totalPairs);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelElapsed = 0;
    cudaEventElapsedTime(&kernelElapsed, startEvent, stopEvent);
    double kTime = kernelElapsed / 1000.0;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaDeviceSynchronize();

    // Transfer GPU results back to CPU (included in computeTime)
    cudaMemcpy(tsp.populationFlat.data(), tsp.d_population, tsp.populationFlat.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsp.parentFitnessFlat.data(), tsp.d_populationFitness, tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyDeviceToHost);
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - total_start).count();
    // CPU mapping phase: update population structure (not included in computeTime)
    auto mapping_start = high_resolution_clock::now();
    int offset = 0;
    int fit_idx = 0;
    for (int island = 0; island < tsp.numIslands; island++) {
        for (auto &ind : tsp.population[island]) {
            for (int j = 0; j < tsp.numCities; j++) {
                ind.chromosome[j] = tsp.populationFlat[offset + j];
            }
            ind.fitness = tsp.parentFitnessFlat[fit_idx++];
            offset += tsp.numCities;
        }
    }
    auto mapping_end = high_resolution_clock::now();
    double mappingTime = duration_cast<duration<double>>(mapping_end - mapping_start).count();
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    tsp.replacementTime.computeTime += compTime;
    tsp.replacementTime.kernelTime += kTime;
    tsp.replacementTime.totalTime += totTime;
}

// migrationCUDA: Calls the CPU version of migration and records total time (all included)
void migrationCUDA(TSP &tsp) {
    auto total_start = high_resolution_clock::now();
    auto compute_start = high_resolution_clock::now();
    migrationCPU(tsp);
    auto compute_end = high_resolution_clock::now();
    double compTime = duration_cast<duration<double>>(compute_end - compute_start).count();
    auto mapping_start = high_resolution_clock::now();
    tsp.flattenPopulationToHost();
    auto mapping_end = high_resolution_clock::now();
    double mappingTime = duration_cast<duration<double>>(mapping_end - mapping_start).count();
    auto total_end = high_resolution_clock::now();
    double totTime = duration_cast<duration<double>>(total_end - total_start).count();
    tsp.migrationTime.computeTime += compTime;
    tsp.migrationTime.kernelTime += 0;
    tsp.migrationTime.totalTime += totTime;
}

} // namespace GA
