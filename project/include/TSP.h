#ifndef TSP_H
#define TSP_H

#include <vector>
// 定义一个时间记录结构体
struct TimeRecord {
    double computeTime; // 核心计算（CPU 逻辑）时间，不包括数据同步或传输时间
    double kernelTime;  // CUDA 内核执行时间（仅适用于采用 CUDA 的步骤）
    double totalTime;   // 该环节总时间（包括数据传输、同步等）
};

// City structure: represents a city with an ID and coordinates.
struct City {
    int id;
    float x;
    float y;
};

// Individual structure: represents a tour (a permutation of city indices)
// and its associated fitness value and island identifier.
struct Individual {
    std::vector<int> chromosome;  // Order of cities visited
    float fitness;                // Fitness value (e.g., 1 / total distance)
    int islandID;                 // Identifier of the island (subpopulation) this individual belongs to
};

// Define ParentPairs and Offspring types to be consistent with the original GAInterface aliases.
typedef std::vector<std::vector<std::pair<Individual, Individual>>> ParentPairs;
typedef std::vector<std::vector<Individual>> Offspring;

class TSP {
public:
    int numCities;             // Number of cities
    int popSize;               // Total population size (all individuals across islands)
    int mapSize;               // Map size (e.g., if 100, cities are in the range [0, 100])
    int numIslands;            // Number of islands (subpopulations)
    float parentSelectionRate; // Parent selection rate (optional parameter)
    float crossoverProbability;// Crossover probability
    float mutationProbability; // Mutation probability


    // ------------------
    // Original Data Structures
    // ------------------
    std::vector<City> cities;                         // Coordinates of the cities
    std::vector<std::vector<Individual>> population;  // Population divided by islands
    std::vector<std::vector<float>> distanceMatrix;   // 2D matrix representing distances between cities

    // ------------------
    // Additional GA Intermediate Data
    // ------------------
    ParentPairs parentPairs;       // Parent pairs (grouped by island)
    Offspring   offsprings;        // Offspring (grouped by island)

    // ----------------------
    // 各环节计时记录
    // ----------------------
    TimeRecord selectionTime;
    TimeRecord crossoverTime;
    TimeRecord mutationTime;
    TimeRecord replacementTime;
    TimeRecord migrationTime;
    TimeRecord updatePopulationFitnessTime;
    TimeRecord updateOffspringFitnessTime;

    // To facilitate sharing of flattened data between GPU and CPU implementations,
    // additional flattened arrays are maintained:
    // 1) Flattened population data (all individuals in a single continuous array)
    std::vector<int> populationFlat;
    // 2) Flattened distance matrix data (stored in row-major order)
    std::vector<float> distanceMatrixFlat;
    // 3) Flattened parent pair data: chromosomes for parent A and parent B, along with their fitness values.
    std::vector<int> parentAFlat;
    std::vector<int> parentBFlat;
    std::vector<float> parentFitnessFlat; // Flattened fitness values for parent individuals
    // 4) Flattened offspring data: chromosomes and fitness values for offspring.
    std::vector<int> offspringFlat;
    std::vector<float> offspringFitnessFlat;

    // Number of parent pairs per island (used in GPU kernels to determine parallel workload)
    std::vector<int> parentPairCount;

    // ------------------
    // GPU Device Pointers (for persistent allocation, if needed)
    // ------------------
    float *d_distanceMatrix = nullptr;
    int   *d_population     = nullptr;
    float *d_populationFitness = nullptr;

    // Device pointers for parent pairs / children used in CUDA kernels
    int   *d_parentA = nullptr;
    int   *d_parentB = nullptr;
    float *d_parentFitness = nullptr;
    int   *d_child1 = nullptr;
    int   *d_child2 = nullptr;

    // Device pointers for offspring used in mutation/crossover operations
    int   *d_offspring = nullptr;
    float *d_offspringFitness = nullptr;

    // Device pointers used for replacement operations (combining parent and offspring data)
    int   *d_parentChromosomes = nullptr;
    int   *d_offspringChromosomes = nullptr;

    // Constructor: initializes the TSP instance with the given parameters:
    // number of cities, population size, map size, number of islands,
    // parent selection rate, crossover probability, and mutation probability.
    TSP(int _numCities, int _popSize, int _mapSize, int _numIslands,
        float _parentSelectionRate, float _crossoverProbability, float _mutationProbability);

    // Initialization functions for cities, population, and distance matrix.
    void initCities();
    void initPopulation();
    void computeDistanceMatrix();

    // Functions to flatten the population and distance matrix to one-dimensional arrays
    // and to allocate GPU buffers using cudaMalloc.
    void flattenPopulationToHost();
    void flattenDistanceMatrixToHost();
    void allocateGPUBuffers();

    // Functions to copy flattened data from host to GPU, if the population is updated.
    void copyPopulationHostToDevice();
    void copyDistanceMatrixHostToDevice();

    // Destructor: releases all allocated GPU memory.
    ~TSP();
};

#endif
