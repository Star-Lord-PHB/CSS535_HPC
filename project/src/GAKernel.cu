#include "GAKernel.hpp"
#include <stdexcept>
#include <curand_kernel.h>


struct Array2D {

    int* array;
    long x, y;

    Array2D(int* array, long x, long y): array(array), x(x), y(y) {}

    int* operator[](long i) const {
        if (i < 0 || i > x) {
            throw std::runtime_error("array index out of bound");
        }
        return array + i * y; 
    } 

};


struct Array3D {

    int* array;
    long x, y, z;

    Array3D(int* array, long x, long y, long z): array(array), x(x), y(y), z(z) {}

    Array2D operator[](long i) const {
        if (i < 0 || i > x) {
            throw std::runtime_error("array index out of bound");
        }
        return {array + i * y * z, y, z};
    }

    [[nodiscard]] constexpr size_t totalSize() const noexcept {
        return x * y * z;
    }

    [[nodiscard]] int at(long i, long j, long k) const {
        const auto index = (size_t) i * y * z + j * z + k;
        if (totalSize() <= index) {
            throw std::runtime_error("array index out of bound");
        }
        return array[index];
    }

};


__device__ int calculate_fitness_kernel(int* individual, int* fitnessCalcSpace) {

}



__global__ void ga_kernel(int* const solution, const int geneMin, int const geneMax) {

    // layout of the shared memory:
    // [------- chromosome -------][fitness][------- offspring -------][fitness][------- fitness calc -------][sharedRandomValue]
    // total length: chromosomeLength * 3 + 3
    extern  __shared__ int sharedMemory[];

    const auto islandCount = gridDim.x;
    const auto individualCount = gridDim.y;
    const auto chromosomeLength = blockDim.x;

    const auto islandId = blockIdx.x;
    const auto individualId = blockIdx.y;
    const auto geneId = threadIdx.x;

    const auto threadId = islandId * individualCount * chromosomeLength + individualId * chromosomeLength + geneId;
    const auto globalGeneId = islandId * individualCount * (chromosomeLength + 1) + individualId * (chromosomeLength + 1) + geneId;

    // set up the shared memory layout
    auto const localChromosome = sharedMemory;
    auto const localChromosomeFitness = localChromosome + chromosomeLength;
    auto const offspring = sharedMemory + chromosomeLength + 1;
    auto const offspringFitness = offspring + chromosomeLength;
    auto const fitnessCalcArea = sharedMemory + chromosomeLength * 2;
    auto const blockSharedRandomValue = (float*) sharedMemory + chromosomeLength * 3 + 2;

    // operating on either `solution` or `solutionWrapped` are the same 
    auto solutionWrapped = Array3D(solution, islandCount, individualCount, chromosomeLength);

    curandState curandState;
    curand_init(42, threadId, 0, &curandState);

    {
        // copying chromosome to shared memory
        localChromosome[geneId] = solution[globalGeneId];
        if (geneId == chromosomeLength - 1) {
            *localChromosomeFitness = solution[globalGeneId] + 1;
        }
    }

    __syncthreads();

    {
        // crossover
        if (geneId == 0) {
            // the first thread in the block deside whether to do the crossover
            *blockSharedRandomValue = curand_uniform(&curandState);
        }
        __syncthreads();

        // Generate offspring
        if (*blockSharedRandomValue > 0.7) {
            // randomly choose the other parent
            const auto pairedIndividualId = (unsigned int)floor(curand_uniform(&curandState) * individualCount);
            // randomly choose the crossover point
            const auto crossoverPoint = (unsigned int)floor(curand_uniform(&curandState) * (chromosomeLength - 1)) + 1;
            // Generate the offspring
            // TODO: Check whether this is really better than if-else
            offspring[geneId] = (geneId >= crossoverPoint) ? solutionWrapped.at(islandId, pairedIndividualId, geneId) : localChromosome[geneId];
        } else {
            // Generate the offspring
            offspring[geneId] = localChromosome[geneId];
        }
    }

    __syncthreads();

    {
        // mutation
        const auto mutationTriggerValue = curand_uniform(&curandState);
        offspring[geneId] = (mutationTriggerValue > 0.95) ? (floor(curand_uniform(&curandState) * (geneMax - geneMin)) + geneMin) : offspring[geneId];
    }

    __syncthreads();

    {
        // Calculate fitness
        // auto counter = chromosomeLength + 1;
        // auto threadCounter = geneId;
        // if (counter > 1 && threadCounter % 2 == 0) {
        //     // in the first iteration, need to copy data from `offspring` to `fitnessCalcArea`, so do it separately
        //     fitnessCalcArea[geneId] = (threadCounter + 1 == counter) ? offspring[geneId] : min(offspring[geneId], offspring[geneId + 1]);
        //     counter /= 2;
        //     threadCounter /= 2;
        // }
        // __syncthreads();
        // for (; counter > 1; counter /= 2) {
        //     if (threadCounter % 2 == 0) {
        //         fitnessCalcArea[geneId] = (threadCounter + 1 == counter) ? fitnessCalcArea[geneId] : min(fitnessCalcArea[geneId], fitnessCalcArea[geneId + 1]);
        //         threadCounter /= 2;
        //     }
        //     __syncthreads();
        // }
        // if (geneId == 0) {
        //     *offspringFitness = fitnessCalcArea[0];
        // }
        *offspringFitness = calculate_fitness_kernel(offspring, fitnessCalcArea);
    }

    {
        // Selection
        if (*offspringFitness > *localChromosomeFitness) {
            localChromosome[geneId] = offspring[geneId];
            solution[globalGeneId] = offspring[geneId];
            if (geneId == chromosomeLength - 1) {
                *localChromosomeFitness = *offspringFitness;
                solution[globalGeneId + 1] = *offspringFitness;
            }
        }
    }

}