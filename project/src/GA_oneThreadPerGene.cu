#include "GA_oneThreadPerGene.hpp"
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>


/// @brief 2D array used in CUDA kernel, the underlying storage is still 1D
template<typename  T>
struct CudaArray2D {

    T* array;
    long x, y;

     __device__ CudaArray2D(T* array, long x, long y): array(array), x(x), y(y) {}

    __device__ const T& at(long i, long j) const {
         return array[i * y + j];
    }

    __device__ T& at(long i, long j) {
         return array[i * y + j];
    }

    __device__ T* operator[](long i) const {
        return array + i * y; 
    } 

};


/// @brief 3D array used in CUDA kernel, the underlying storage is still 1D
template<typename  T>
struct CudaArray3D {

    T* array;
    long x, y, z;

    __device__ CudaArray3D(T* array, long x, long y, long z): array(array), x(x), y(y), z(z) {}

    __device__ CudaArray2D<T> operator[](long i) const {
        return { array + i * y * z, y, z };
    }

    __device__ [[nodiscard]] constexpr size_t totalSize() const noexcept {
        return x * y * z;
    }

    __device__ [[nodiscard]] const T& at(long i, long j, long k) const {
        return array[i * y * z + j * z + k];
    }

    __device__ [[nodiscard]] T& at(long i, long j, long k) {
        return array[i * y * z + j * z + k];
    }

};


/// @brief Metadata of a CUDA thread, including a thread local randd state and a block shared one
struct KernelSpec {
    const unsigned int islandCount;
    const unsigned int individualCount;
    const unsigned int chromosomeLength;
    const unsigned int islandId;
    const unsigned int individualId;
    const unsigned int geneId;
    const unsigned int threadId;
    const unsigned int globalGeneId;
    const unsigned int iteration;
    curandState randState;
    curandState blockSharedRandState;
    __device__ KernelSpec(
        const unsigned int islandCount,
        const unsigned int individualCount,
        const unsigned int chromosomeLength,
        const unsigned int islandId,
        const unsigned int individualId,
        const unsigned int geneId,
        const unsigned int iteration = 0,
        const curandState& randState = curandState(),
        const curandState& blockSharedRandState = curandState()
    ):
    islandCount(islandCount), individualCount(individualCount), chromosomeLength(chromosomeLength),
    islandId(islandId), individualId(individualId), geneId(geneId), iteration(iteration),
    threadId(islandId * individualCount * chromosomeLength + individualId * chromosomeLength + geneId),
    globalGeneId(islandId * individualCount * (chromosomeLength + 1) + individualId * (chromosomeLength + 1) + geneId),
    randState(randState), blockSharedRandState(blockSharedRandState) {}
};


/// @brief swap two elements
template<typename T>
__device__ void cuda_swap(T& a, T& b) {
    const auto temp = a;
    a = b;
    b = temp;
}


/// @brief calculate the fitness of the input individual and place the output in `output`
__device__ void calculate_fitness(
    const int* const individual,
    float* const fitnessCalcSpace,
    const CudaArray2D<float>& distanceMat,
    const KernelSpec& spec,
    float* const output
) {

    auto geneId = spec.geneId;
    fitnessCalcSpace[geneId] = distanceMat.at(individual[geneId], individual[(geneId + 1) % spec.chromosomeLength]);

    __syncthreads();

    for (auto step = 1; step < spec.chromosomeLength; step <<= 1) {
        if (spec.geneId % (step << 1) == 0) {
            auto pairGeneId = spec.geneId ^ step;
            if (pairGeneId < spec.chromosomeLength) {
                fitnessCalcSpace[spec.geneId] += fitnessCalcSpace[pairGeneId];
            }
        }
        __syncthreads();
    }

    if (spec.geneId == 0) {
        *output = fitnessCalcSpace[0];
    }

}



/// @brief create one random chromosome
__device__ void make_initial_chromosome(int* const chromosome, float* const randomArray, KernelSpec& spec) {

    chromosome[spec.geneId] = static_cast<int>(spec.geneId);
    randomArray[spec.geneId] = curand_uniform(&spec.randState);

    __syncthreads();

    for (unsigned stage = 2; stage < spec.chromosomeLength; stage <<= 1 /* stage *= 2 */) {

        for (unsigned step = stage >> 1; step > 0; step >>= 1 /* step /= 2 */) {

            const auto paired = spec.geneId ^ step;

            if (paired > spec.geneId && paired < spec.chromosomeLength) {

                const auto ascending = (spec.geneId & stage) == 0;

                const auto value1 = randomArray[spec.geneId];
                const auto value2 = randomArray[paired];

                if (value1 > value2 == ascending) {
                    randomArray[paired] = value1;
                    randomArray[spec.geneId] = value2;
                    cuda_swap(chromosome[spec.geneId], chromosome[paired]);
                }

            }

            __syncthreads();

        }

    }

}



/// @brief create random initialized solution
__global__ void ga_kernel_initialize(int* const solution, float* const distanceMat) {

    // layout:
    // [------- chromosome -------][fitness][------- local workspace -------]
    // total length: chromosomeLength * 2 + 1
    extern  __shared__ int sharedMemory[];

    auto spec = KernelSpec(gridDim.x, gridDim.y, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);
    curand_init(42, spec.threadId, 0, &spec.randState);

    auto const localChromosome = sharedMemory;
    auto const localChromosomeFitness = reinterpret_cast<float*>(localChromosome + spec.chromosomeLength);
    auto const localWorkspace = reinterpret_cast<float*>(sharedMemory + spec.chromosomeLength + 1);

    const auto distanceMatWrapped = CudaArray2D(distanceMat, spec.chromosomeLength, spec.chromosomeLength);

    {
        // Initialize the solution

        // Create random initial chromosome
        make_initial_chromosome(localChromosome, localWorkspace, spec);
        // Calculate the fitness of the initial chromosome
        calculate_fitness(localChromosome, localWorkspace, distanceMatWrapped, spec, localChromosomeFitness);

        __syncthreads();

        // Sync the initial solution to global space
        solution[spec.globalGeneId] = localChromosome[spec.geneId];
        if (spec.geneId == spec.chromosomeLength - 1) {
            solution[spec.globalGeneId + 1] = *reinterpret_cast<int*>(localChromosomeFitness);
        }

    }

}



/// @brief PMX crossover
__device__ void ga_one_thread_per_gene_crossover(
    const CudaArray3D<int>& solution,
    const int* const individual,
    const float crossoverProb,
    int* const mappingSpace,
    int* const offspringOut,
    KernelSpec& spec
) {

    const auto crossoverTrigger = curand_uniform(&spec.blockSharedRandState);
    // randomly choose the other parent
    const auto pairedIndividualId = static_cast<unsigned int>(floor(curand_uniform(&spec.blockSharedRandState) * static_cast<float>(spec.individualCount)));
    // randomly choose the crossover point
    const auto crossoverPoint1 = static_cast<unsigned int>(floor(curand_uniform(&spec.blockSharedRandState) * static_cast<float>(spec.chromosomeLength - 2)));
    const auto crossoverPoint2 = static_cast<unsigned int>(floor(curand_uniform(&spec.blockSharedRandState) * static_cast<float>(spec.chromosomeLength - crossoverPoint1 - 1))) + crossoverPoint1;

    mappingSpace[spec.geneId] = -1;

    __syncthreads();

    if (crossoverTrigger < crossoverProb) {

        if (spec.geneId >= crossoverPoint1 && spec.geneId <= crossoverPoint2) {

            auto srcValue = individual[spec.geneId];
            auto destValue = solution.at(spec.islandId, pairedIndividualId, spec.geneId);

            offspringOut[spec.geneId] = destValue;
            mappingSpace[destValue] = srcValue;

        }

        __syncthreads();

        if (spec.geneId < crossoverPoint1 || spec.geneId > crossoverPoint2) {
            const auto original = individual[spec.geneId];
            auto mappedValue = mappingSpace[original];
            while (mappedValue != -1 && mappingSpace[mappedValue] != -1) {
                mappedValue = mappingSpace[mappedValue];
            }
            offspringOut[spec.geneId] = mappedValue == -1 ? original : mappedValue;
        }

    } else {

        offspringOut[spec.geneId] = individual[spec.geneId];
        __syncthreads();

    }

}



/// @brief Insertion Mutation
__device__ void ga_one_thread_per_gene_mutation(
    int* const individual,
    const float mutationProb,
    KernelSpec& spec
) {

    const auto insertSrcId = static_cast<unsigned int>(floor(curand_uniform(&spec.blockSharedRandState) * static_cast<float>(spec.chromosomeLength - 1)));
    const auto insertDestId = static_cast<unsigned int>(floor(curand_uniform(&spec.blockSharedRandState) * static_cast<float>(spec.chromosomeLength - 1)));
    const auto mutationTrigger = curand_uniform(&spec.blockSharedRandState);

    const auto value = individual[spec.geneId];

    __syncthreads();

    if (mutationTrigger < mutationProb && insertSrcId != insertDestId) {

        if (spec.geneId == insertSrcId) {
            individual[insertDestId] = value;
        } else {
            const auto upperBound = max(insertSrcId, insertDestId);
            const auto lowerBound = min(insertSrcId, insertDestId);
            if (spec.geneId >= lowerBound && spec.geneId <= upperBound) {
                const auto destId = spec.geneId + ((insertDestId < insertSrcId) ? 1 : -1);
                individual[destId] = value;
            }
        }

    }

}



/// @brief Crossover + Mutation + Fitness Calculation + Replacement
__global__ void ga_one_thread_per_gene_kernel1(
    int* const solution,
    float* const distanceMat,
    int* const solutionGlobalWorkspace,
    const float crossoverProb,
    const float mutationProb,
    const unsigned iteration
) {

    // layout of the shared memory:
    // [------- chromosome -------][fitness][------- offspring -------][fitness][------- local workspace -------]
    // total length: chromosomeLength * 3 + 2
    extern  __shared__ int sharedMemory[];

    auto spec = KernelSpec(gridDim.x, gridDim.y, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x, iteration);
    curand_init(42, spec.threadId, iteration * 10, &spec.randState);
    curand_init(42, spec.islandId * spec.individualCount + spec.individualId, iteration * 10, &spec.blockSharedRandState);

    // set up the shared memory layout
    auto const localChromosome = sharedMemory;
    auto const localChromosomeFitness = reinterpret_cast<float*>(localChromosome + spec.chromosomeLength);
    auto const offspring = sharedMemory + spec.chromosomeLength + 1;
    auto const offspringFitness = reinterpret_cast<float*>(offspring + spec.chromosomeLength);
    auto const localWorkspace = reinterpret_cast<float*>(sharedMemory + spec.chromosomeLength * 2 + 2);

    // operating on either `solution` or `solutionWrapped` are the same 
    auto solutionWrapped = CudaArray3D(solution, spec.islandCount, spec.individualCount, spec.chromosomeLength + 1);
    auto distanceMatWrapped = CudaArray2D(distanceMat, spec.chromosomeLength, spec.chromosomeLength);

    {
        // Copy the individual from global space to shared memory
        localChromosome[spec.geneId] = solution[spec.globalGeneId];
        if (spec.geneId == spec.chromosomeLength - 1) {
            *localChromosomeFitness = *reinterpret_cast<float*>(&solution[spec.globalGeneId + 1]);
        }
    }

    __syncthreads();

    {
        // crossover
        ga_one_thread_per_gene_crossover(
            solutionWrapped, localChromosome, crossoverProb, reinterpret_cast<int*>(localWorkspace), offspring, spec
        );
    }

    __syncthreads();

    {
        // mutation
        ga_one_thread_per_gene_mutation(offspring, mutationProb, spec);
    }

    __syncthreads();

    {
        // Calculate fitness
        calculate_fitness(offspring, localWorkspace, distanceMatWrapped, spec, offspringFitness);
    }

    __syncthreads();

    {
        // Replacement & copy to global workspace
        const auto doReplace = *offspringFitness < *localChromosomeFitness;
        auto const individual = doReplace ? offspring : localChromosome;
        auto const fitness = doReplace ? offspringFitness : localChromosomeFitness;
        solutionGlobalWorkspace[spec.globalGeneId] = individual[spec.geneId];
        if (spec.geneId == spec.chromosomeLength - 1) {
            solutionGlobalWorkspace[spec.globalGeneId + 1] = *reinterpret_cast<int*>(fitness);
        }
    }

}



/// @brief Find the best / worst individual in each island
__global__ void ga_one_thread_per_gene_kernel2(
    const int* const solution,
    const unsigned chromosomeLength,
    unsigned* const bestWorstIndividualIds
) {

    // layout:
    // [------- fitness sorting space -------][------- index space -------]
    // total size: individual count * 2
    extern __shared__ int _sharedMemory[];
    auto const sharedMemory = reinterpret_cast<unsigned*>(_sharedMemory);

    const auto islandCount = gridDim.x;
    const auto individualCount = blockDim.x;
    const auto islandId = blockIdx.x;
    const auto individualId = threadIdx.x;

    auto const fitnessSortingSpace = reinterpret_cast<float*>(sharedMemory);
    auto const indexSpace = sharedMemory + individualCount;
    auto const worstIndividualId = indexSpace;
    auto const bestIndividualId = indexSpace + 1;

    const auto solutionWrapped = CudaArray3D(solution, islandCount, individualCount, chromosomeLength + 1);

    {
        // Find the best / worst individual

        fitnessSortingSpace[individualId] = *reinterpret_cast<const float*>(&(solutionWrapped.at(islandId, individualId, chromosomeLength)));
        indexSpace[individualId] = individualId;

        __syncthreads();

        for (unsigned step = 1; step < individualCount; step <<= 1 /* step *= 2 */) {

            if (individualId % (step << 1) == 0) {

                const auto pairIndividualId = individualId ^ step;

                if (pairIndividualId > individualId && pairIndividualId < individualCount) {

                    const auto value1 = fitnessSortingSpace[individualId];
                    const auto value2 = fitnessSortingSpace[pairIndividualId];

                    if (value1 < value2) {
                        fitnessSortingSpace[individualId] = value2;
                        fitnessSortingSpace[pairIndividualId] = value1;
                        cuda_swap(indexSpace[individualId], indexSpace[pairIndividualId]);
                    }

                }

            }

            __syncthreads();

        }

        const auto minFitnessSortingSpace = fitnessSortingSpace + 1;
        const auto minFitnessIndexSpace = indexSpace + 1;

        for (unsigned step = 1; step < individualCount - 1; step <<= 1 /* step *= 2 */) {

            if (individualId % (step << 1) == 0) {

                const auto pairIndividualId = individualId ^ step;

                if (pairIndividualId > individualId && pairIndividualId < individualCount - 1) {

                    const auto value1 = minFitnessSortingSpace[individualId];
                    const auto value2 = minFitnessSortingSpace[pairIndividualId];

                    if (value1 > value2) {
                        minFitnessSortingSpace[individualId] = value2;
                        minFitnessSortingSpace[pairIndividualId] = value1;
                        const auto temp = minFitnessIndexSpace[individualId];
                        minFitnessIndexSpace[individualId] = minFitnessIndexSpace[pairIndividualId];
                        minFitnessIndexSpace[pairIndividualId] = temp;
                    }

                }

            }

            __syncthreads();

        }

        bestWorstIndividualIds[islandId * 2] = *bestIndividualId;
        bestWorstIndividualIds[islandId * 2 + 1] = *worstIndividualId;

    }

}



/// @brief Migration
__global__ void ga_one_thread_per_gene_kernel3(
    int* const solution,
    const unsigned* const bestWorstIndividualIds,
    const unsigned individualCount
) {

    const auto islandId = blockIdx.x;
    const auto geneId = threadIdx.x;
    const auto chromosomeLength = blockDim.x;
    const auto islandCount = gridDim.x;

    auto solutionWrapped = CudaArray3D(solution, islandCount, individualCount, chromosomeLength);
    auto bestWorstIndividualIdsWrapped = CudaArray2D(bestWorstIndividualIds, islandCount, 2);

    const auto srcIslandId = (islandId + islandCount - 1) % islandCount;

    const auto srcIndividualId = bestWorstIndividualIdsWrapped.at(srcIslandId, 0);
    const auto individualToReplaceId = bestWorstIndividualIdsWrapped.at(islandId, 1);

    solutionWrapped.at(islandId, individualToReplaceId, geneId) = solutionWrapped.at(srcIslandId, srcIndividualId, geneId);

}



/// @brief print 3D cuda array
template<typename T>
void printCudaArray3D(const T* const cuArray, unsigned x, unsigned y, unsigned z) {
    auto array = new Array3D<T>(x, y, z);
    cudaMemcpy(array->data(), cuArray, array->byteCount(), cudaMemcpyDeviceToHost);
    std::cout << *array << std::endl << std::endl;
    delete array;
}


/// @brief print 2D cuda array
template<typename T>
void printCudaArray2D(const T* const cuArray, unsigned x, unsigned y) {
    auto array = new Array2D<T>(x, y);
    cudaMemcpy(array->data(), cuArray, array->byteCount(), cudaMemcpyDeviceToHost);
    std::cout << *array << std::endl << std::endl;
    delete array;
}



/// @brief one thread per gene implementation (GPU)
void ga_one_thread_per_gene(
    Array3D<int>& solution,
    const Array2D<float>& distanceMap,
    const int generation,
    const float crossoverProb,
    const float mutationProb,
    milliseconds* const calculationTime,
    milliseconds* const totalTime
) {

    auto chromosomeLength = solution.zSize() - 1;
    auto individualCount = solution.ySize();
    auto islandCount = solution.xSize();

    const auto solutionSize = islandCount * individualCount * (chromosomeLength + 1);

    int* cudaSolution = nullptr;
    float* cudaDistanceMat = nullptr;
    int* cudaSolutionGlobalWorkspace = nullptr;
    int* cudaMigrationSpace = nullptr;
    float* cudaFitnessSortingSpace = nullptr;
    unsigned* cudaBestWorstIndividualIds = nullptr;

    const auto totalStart = high_resolution_clock::now();

    cudaMalloc(&cudaSolution, solutionSize * sizeof(int));
    cudaMalloc(&cudaSolutionGlobalWorkspace, solutionSize * sizeof(int));
    cudaMalloc(&cudaDistanceMat, distanceMap.byteCount());
    cudaMalloc(&cudaMigrationSpace, islandCount * (chromosomeLength + 1) * sizeof(int));
    cudaMalloc(&cudaFitnessSortingSpace, islandCount * individualCount * sizeof(float));
    cudaMalloc(&cudaBestWorstIndividualIds, islandCount * 2 * sizeof(unsigned));

    cudaMemcpy(cudaDistanceMat, distanceMap.data(), distanceMap.byteCount(), cudaMemcpyHostToDevice);

    const auto calcStart = high_resolution_clock::now();

    size_t sharedMemorySize = (chromosomeLength * 2 + 1) * sizeof(int);
    ga_kernel_initialize<<<dim3(islandCount, individualCount, 1), chromosomeLength, sharedMemorySize>>>(cudaSolution, cudaDistanceMat);
    cudaDeviceSynchronize();

    // printCudaArray3D(cudaSolution, islandCount, individualCount, chromosomeLength + 1);

    for (unsigned i = 0; i < generation; i++) {

        // std::cout << "iteration " << i << std::endl;

        sharedMemorySize = (chromosomeLength * 3 + 2) * sizeof(int);
        ga_one_thread_per_gene_kernel1<<<dim3(islandCount, individualCount, 1), chromosomeLength, sharedMemorySize>>>(
            cudaSolution,
            cudaDistanceMat,
            cudaSolutionGlobalWorkspace,
            crossoverProb,
            mutationProb,
            i
        );
        cudaDeviceSynchronize();

        cudaMemcpy(cudaSolution, cudaSolutionGlobalWorkspace, solutionSize * sizeof(int), cudaMemcpyDeviceToDevice);

        // {
        //     auto solution1 = new Array3D<int>(islandCount, individualCount, chromosomeLength + 1);
        //     cudaMemcpy(solution1->data(), cudaSolution, solution1->byteCount(), cudaMemcpyDeviceToHost);
        //     for (auto i = 0; i < islandCount; i++) {
        //         for (auto j = 0; j < individualCount; j++) {
        //             auto sum = 0;
        //             for (auto k = 0; k < chromosomeLength; k++) {
        //                 sum += solution1->at(i, j, k);
        //             }
        //             if (sum != 32640) {
        //                 std::cout << "sum: " << sum << std::endl;
        //                 std::cout << i << " " << j << std::endl;
        //                 for (auto k = 0; k < chromosomeLength; k++) {
        //                     std::cout << solution1->at(i, j, k) << " ";
        //                 }
        //                 std::cout << std::endl;
        //                 exit(1);
        //             }
        //         }
        //     }
        //     delete solution1;
        // }

        // printCudaArray3D(cudaSolution, islandCount, individualCount, chromosomeLength + 1);

        sharedMemorySize = individualCount * 2 * sizeof(int);
        ga_one_thread_per_gene_kernel2<<<islandCount, individualCount, sharedMemorySize>>>(
            cudaSolution,
            chromosomeLength,
            cudaBestWorstIndividualIds
        );
        cudaDeviceSynchronize();

        // printCudaArray2D(cudaBestWorstIndividualIds, islandCount, 2);

        ga_one_thread_per_gene_kernel3<<<islandCount, chromosomeLength + 1>>>(
            cudaSolution,
            cudaBestWorstIndividualIds,
            individualCount
        );
        cudaDeviceSynchronize();

        // printCudaArray3D(cudaSolution, islandCount, individualCount, chromosomeLength + 1);

    }

    const auto calcEnd = high_resolution_clock::now();

    cudaMemcpy(solution.data(), cudaSolution, solution.byteCount(), cudaMemcpyDeviceToHost);

    const auto totalEnd = high_resolution_clock::now();

    *calculationTime = duration_cast<milliseconds>(calcEnd - calcStart);
    *totalTime = duration_cast<milliseconds>(totalEnd - totalStart);

    cudaFree(cudaSolution);
    cudaFree(cudaDistanceMat);
    cudaFree(cudaMigrationSpace);
    cudaFree(cudaFitnessSortingSpace);
    cudaFree(cudaSolutionGlobalWorkspace);

}


void ga_one_thread_per_gene_test() {

    auto solution = Array3D<int>(2, 3, 6);
    const auto distanceMap = Array2D<float>({
        { 1, 2, 4, 3, 5 },
        { 2, 1, 5, 4, 3 },
        { 4, 5, 1, 2, 1 },
        { 3, 4, 2, 1, 4 },
        { 5, 3, 1, 4, 1 },
    });
    auto time1 = milliseconds();
    auto time2 = milliseconds();

    ga_one_thread_per_gene(solution, distanceMap, 1, 1, 1, &time1, &time2);

    std::cout << solution << std::endl;

}