// GA_CUDA.cu
#include "GA_CUDA.h"
#include "GA_CPU.h"  // For reference: partial use of CPU functions (only for selectionCUDA)
#include "TSP.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>
#include <iostream>

namespace GA {

    // -----------------------------------------------------------
    // Kernel: Compute fitness for each individual
    // -----------------------------------------------------------
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
            // Add distance from last city back to the first city
            int lastCity = d_population[base + numCities - 1];
            int firstCity = d_population[base];
            totalDist += d_distanceMatrix[lastCity * numCities + firstCity];
            d_fitness[idx] = (totalDist <= 0.0f) ? 0.0f : (1.0f / totalDist);
        }
    }

    // -----------------------------------------------------------
    // Kernel: Order Crossover (OX)
    // Each thread processes one pair of parents.
    // -----------------------------------------------------------
    __global__ void crossoverKernel(const int *d_parentA, const int *d_parentB,
                                    int *d_child1, int *d_child2,
                                    int numPairs, int numCities,
                                    float crossoverProb, unsigned long seed)
    {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx >= numPairs) return;

        // Initialize curand state for each thread
        curandState state;
        curand_init(seed, pairIdx, 0, &state);

        int base = pairIdx * numCities;
        int *child1 = d_child1 + base;
        int *child2 = d_child2 + base;

        float r = curand_uniform(&state);
        if (r >= crossoverProb) {
            // No crossover: directly copy the parent's chromosomes
            for (int i = 0; i < numCities; i++) {
                child1[i] = d_parentA[base + i];
                child2[i] = d_parentB[base + i];
            }
            return;
        }

        // Initialize children chromosomes with -1 (indicating unfilled positions)
        for (int i = 0; i < numCities; i++) {
            child1[i] = -1;
            child2[i] = -1;
        }
        // Randomly select two crossover points
        int p1 = curand(&state) % numCities;
        int p2 = curand(&state) % numCities;
        if (p1 > p2) {
            int tmp = p1; p1 = p2; p2 = tmp;
        }
        // Copy the crossover segment from the parents
        for (int i = p1; i <= p2; i++) {
            child1[i] = d_parentA[base + i];
            child2[i] = d_parentB[base + i];
        }
        // Fill child1: take genes from parentB in order
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
        // Fill child2: take genes from parentA in order
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

    // -----------------------------------------------------------
    // Kernel: Mutation
    // Each thread processes one individual.
    // -----------------------------------------------------------
    __global__ void mutationKernel(int *d_offspring, int totalIndividuals, int numCities,
                                   float mutationProb, unsigned long seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= totalIndividuals) return;

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

    // -----------------------------------------------------------
    // Kernel: Replacement
    // Each thread processes one pair of parents and the corresponding two children.
    // It selects the two best individuals from {parent A, parent B, child1, child2}
    // and replaces the original parents in the population.
    // -----------------------------------------------------------
    __global__ void replacementKernel(
        int *d_population, float *d_populationFitness,
        const int *d_parentA, const int *d_parentB,
        const float *d_parentFitness,
        const int *d_child1, const int *d_child2,
        const float *d_childFitness,
        int numPairs, int numCities)
    {
        int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pairIdx >= numPairs) return;

        int base = pairIdx * numCities;
        // The four individuals (each with chromosome length numCities):
        // Index 0: parent A, Index 1: parent B, Index 2: child1, Index 3: child2.
        float fits[4] = {
            d_parentFitness[2 * pairIdx],
            d_parentFitness[2 * pairIdx + 1],
            d_childFitness[2 * pairIdx],
            d_childFitness[2 * pairIdx + 1]
        };
        const int* chrom[4] = {
            d_parentA + base,
            d_parentB + base,
            d_child1 + base,
            d_child2 + base
        };

        // Find the indices of the two individuals with the highest fitness values
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
        // Replace the corresponding parent's chromosome in the population
        int popOffsetA = pairIdx * 2 * numCities;  // Location of parent A in the flattened population
        int popOffsetB = popOffsetA + numCities;     // Location of parent B in the flattened population
        for (int i = 0; i < numCities; i++) {
            d_population[popOffsetA + i] = chrom[bestIdx][i];
            d_population[popOffsetB + i] = chrom[secondIdx][i];
        }
        // Update the fitness values for these two individuals in the population
        d_populationFitness[2 * pairIdx] = fits[bestIdx];
        d_populationFitness[2 * pairIdx + 1] = fits[secondIdx];
    }

    // -----------------------------------------------------------
    // selectionCUDA
    // Uses the CPU implementation for selection and updates the flattened data in TSP.
    // -----------------------------------------------------------
    ParentPairs selectionCUDA(TSP &tsp) {
        // Use CPU selection and update TSP.parentAFlat, parentBFlat, parentFitnessFlat accordingly
        ParentPairs pp = selectionCPU(tsp);
        tsp.parentPairs = pp;
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += pp[i].size();
        }
        tsp.parentAFlat.resize(totalPairs * tsp.numCities);
        tsp.parentBFlat.resize(totalPairs * tsp.numCities);
        tsp.parentFitnessFlat.resize(2 * totalPairs);
        int pairIndex = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (size_t i = 0; i < pp[island].size(); i++) {
                const Individual &pa = pp[island][i].first;
                const Individual &pb = pp[island][i].second;
                for (int j = 0; j < tsp.numCities; j++) {
                    tsp.parentAFlat[pairIndex * tsp.numCities + j] = pa.chromosome[j];
                    tsp.parentBFlat[pairIndex * tsp.numCities + j] = pb.chromosome[j];
                }
                tsp.parentFitnessFlat[2 * pairIndex]     = pa.fitness;
                tsp.parentFitnessFlat[2 * pairIndex + 1] = pb.fitness;
                pairIndex++;
            }
        }
        return pp;
    }

    // -----------------------------------------------------------
    // crossoverCUDA
    // Uses the pre-flattened TSP.parentAFlat and TSP.parentBFlat without re-flattening.
    // -----------------------------------------------------------
    Offspring crossoverCUDA(TSP &tsp, const ParentPairs &parentPairs) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += parentPairs[i].size();
        }
        // TSP.parentAFlat and parentBFlat have been updated in selectionCUDA
        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        unsigned long seed = time(nullptr);
        // Launch the crossover kernel
        crossoverKernel<<<blocks, threads>>>(tsp.d_parentA, tsp.d_parentB, tsp.d_child1, tsp.d_child2,
                                               totalPairs, tsp.numCities, tsp.crossoverProbability, seed);
        cudaDeviceSynchronize();
        // Copy the kernel results into TSP.offspringFlat (size = totalPairs * 2 * numCities)
        tsp.offspringFlat.resize(totalPairs * 2 * tsp.numCities);
        cudaMemcpy(tsp.offspringFlat.data(), tsp.d_child1, totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tsp.offspringFlat.data() + totalPairs * tsp.numCities, tsp.d_child2,
                   totalPairs * tsp.numCities * sizeof(int), cudaMemcpyDeviceToHost);
        // Reconstruct the Offspring structure (without additional flattening)
        Offspring offspring;
        offspring.resize(tsp.numIslands);
        int pairIndex = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            int numPairs = parentPairs[island].size();
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
                offspring[island].push_back(child1);
                offspring[island].push_back(child2);
                pairIndex++;
            }
        }
        return offspring;
    }

    // -----------------------------------------------------------
    // mutationCUDA
    // Uses the flattened TSP.offspringFlat data without re-flattening in each function.
    // -----------------------------------------------------------
    void mutationCUDA(TSP &tsp, Offspring &offspring) {
        // It is assumed that TSP.offspringFlat has been updated by crossoverCUDA
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int totalOffspring = totalPairs * 2;
        int totalGenes = tsp.offspringFlat.size(); // equals totalOffspring * tsp.numCities
        // Copy TSP.offspringFlat to device (if not already resident)
        cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        unsigned long seed = time(nullptr);
        mutationKernel<<<blocks, threads>>>(tsp.d_offspring, totalOffspring, tsp.numCities, tsp.mutationProbability, seed);
        cudaDeviceSynchronize();
        // Copy the mutated result back to TSP.offspringFlat
        cudaMemcpy(tsp.offspringFlat.data(), tsp.d_offspring, totalGenes * sizeof(int), cudaMemcpyDeviceToHost);
        // Update the Offspring structure directly from the flattened data
        int offset = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                for (int j = 0; j < tsp.numCities; j++) {
                    child.chromosome[j] = tsp.offspringFlat[offset * tsp.numCities + j];
                }
                offset++;
            }
        }
    }

    // -----------------------------------------------------------
    // updateOffspringFitnessCUDA
    // Uses the existing flattened TSP.offspringFlat data.
    // -----------------------------------------------------------
    void updateOffspringFitnessCUDA(TSP &tsp, Offspring &offspring) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += tsp.parentPairs[i].size();
        }
        int totalOffspring = totalPairs * 2;
        int totalGenes = tsp.offspringFlat.size();
        // Copy TSP.offspringFlat to device
        cudaMemcpy(tsp.d_offspring, tsp.offspringFlat.data(), totalGenes * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (totalOffspring + threads - 1) / threads;
        computeFitnessKernel<<<blocks, threads>>>(tsp.d_offspring, tsp.d_distanceMatrix, tsp.d_offspringFitness, tsp.numCities, totalOffspring);
        cudaDeviceSynchronize();
        tsp.offspringFitnessFlat.resize(totalOffspring);
        cudaMemcpy(tsp.offspringFitnessFlat.data(), tsp.d_offspringFitness, totalOffspring * sizeof(float), cudaMemcpyDeviceToHost);
        // Update the fitness values in the Offspring structure
        int idx = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &child : offspring[island]) {
                child.fitness = tsp.offspringFitnessFlat[idx++];
            }
        }
    }

    // -----------------------------------------------------------
    // updatePopulationFitnessCUDA
    // -----------------------------------------------------------
    void updatePopulationFitnessCUDA(TSP &tsp) {
        // Ensure that tsp.populationFlat is up-to-date (should be updated externally if needed)
        cudaMemcpy(tsp.d_population, tsp.populationFlat.data(), tsp.populationFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (tsp.popSize + threads - 1) / threads;
        computeFitnessKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_distanceMatrix, tsp.d_populationFitness, tsp.numCities, tsp.popSize);
        cudaDeviceSynchronize();
        std::vector<float> h_fit(tsp.popSize);
        cudaMemcpy(h_fit.data(), tsp.d_populationFitness, tsp.popSize * sizeof(float), cudaMemcpyDeviceToHost);
        int idx = 0;
        for (int island = 0; island < tsp.numIslands; island++) {
            for (auto &ind : tsp.population[island]) {
                ind.fitness = h_fit[idx++];
            }
        }
    }

    // -----------------------------------------------------------
    // replacementCUDA
    // GPU implementation of replacement: select the best two from {child1, child2, parentA, parentB}
    // to replace the original parents.
    // -----------------------------------------------------------
    void replacementCUDA(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring) {
        int totalPairs = 0;
        for (int i = 0; i < tsp.numIslands; i++) {
            totalPairs += parentPairs[i].size();
        }
        // Assume TSP.parentAFlat and parentBFlat have been updated by selectionCUDA,
        // and TSP.offspringFlat and offspringFitnessFlat have been updated by crossoverCUDA and updateOffspringFitnessCUDA.

        // Copy parent and offspring data to device using the flattened arrays already stored in TSP.
        cudaMemcpy(tsp.d_parentA, tsp.parentAFlat.data(), tsp.parentAFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_parentB, tsp.parentBFlat.data(), tsp.parentBFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_parentFitness, tsp.parentFitnessFlat.data(), tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
        // For offspring, child1 and child2 are stored in the first and second halves of TSP.offspringFlat respectively.
        int totalGenesPerChild = totalPairs * tsp.numCities;
        cudaMemcpy(tsp.d_child1, tsp.offspringFlat.data(), totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tsp.d_child2, tsp.offspringFlat.data() + totalGenesPerChild, totalGenesPerChild * sizeof(int), cudaMemcpyHostToDevice);
        // Copy offspring fitness from TSP.offspringFitnessFlat to device
        cudaMemcpy(tsp.d_offspringFitness, tsp.offspringFitnessFlat.data(), tsp.offspringFitnessFlat.size() * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (totalPairs + threads - 1) / threads;
        replacementKernel<<<blocks, threads>>>(tsp.d_population, tsp.d_populationFitness,
                                                 tsp.d_parentA, tsp.d_parentB, tsp.d_parentFitness,
                                                 tsp.d_child1, tsp.d_child2, tsp.d_offspringFitness,
                                                 totalPairs, tsp.numCities);
        cudaDeviceSynchronize();

        // Copy the updated flattened population data from device back to TSP.populationFlat
        cudaMemcpy(tsp.populationFlat.data(), tsp.d_population, tsp.populationFlat.size() * sizeof(int), cudaMemcpyDeviceToHost);
        // Also copy updated fitness values back to tsp.parentFitnessFlat for further updates
        cudaMemcpy(tsp.parentFitnessFlat.data(), tsp.d_populationFitness, tsp.parentFitnessFlat.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Update TSP.population (assuming the order in populationFlat corresponds to the order in population)
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
    }

    // -----------------------------------------------------------
    // migrationCUDA
    // For migration, we simply call the CPU version as a placeholder.
    // -----------------------------------------------------------
    void migrationCUDA(TSP &tsp) {
        migrationCPU(tsp);
        // Update the flattened population data if necessary
        tsp.flattenPopulationToHost();
    }

} // namespace GA
