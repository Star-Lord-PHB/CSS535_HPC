#include "TSP.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// Compute the Euclidean distance between two cities (internal use)
static float euclideanDistance(const City &a, const City &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Constructor: Initializes the TSP instance with the given parameters
TSP::TSP(int _numCities, int _popSize, int _mapSize, int _numIslands,
         float _parentSelectionRate, float _crossoverProbability, float _mutationProbability)
    : numCities(_numCities),
      popSize(_popSize),
      mapSize(_mapSize),
      numIslands(_numIslands),
      parentSelectionRate(_parentSelectionRate),
      crossoverProbability(_crossoverProbability),
      mutationProbability(_mutationProbability)
{
    srand(static_cast<unsigned>(time(nullptr)));
    // 1) Initialize cities and population
    initCities();
    initPopulation();
    // 2) Compute the distance matrix between cities
    computeDistanceMatrix();
    // 3) Flatten the population and distance matrix on the host side
    flattenPopulationToHost();
    flattenDistanceMatrixToHost();
    // 4) Allocate GPU buffers
    allocateGPUBuffers();
    // Copy the flattened data from host to GPU
    copyPopulationHostToDevice();
    copyDistanceMatrixHostToDevice();
}

// Randomly generate city coordinates and store them in the 'cities' vector
void TSP::initCities() {
    cities.resize(numCities);
    for (int i = 0; i < numCities; i++) {
        City c;
        c.id = i;
        c.x = static_cast<float>(rand() % mapSize);
        c.y = static_cast<float>(rand() % mapSize);
        cities[i] = c;
    }
}

// Initialize the population by evenly dividing the total population among islands
// and randomly shuffling each individual's chromosome.
void TSP::initPopulation() {
    population.resize(numIslands);
    int baseCount = popSize / numIslands;
    int remainder = popSize % numIslands;
    std::random_device rd;
    std::mt19937 g(rd());

    int popAssigned = 0;
    for (int island = 0; island < numIslands; island++) {
        int islandPop = baseCount + (island < remainder ? 1 : 0);
        population[island].resize(islandPop);
        for (int i = 0; i < islandPop; i++) {
            Individual &ind = population[island][i];
            ind.chromosome.resize(numCities);
            // Initialize chromosome with sequential city indices
            for (int j = 0; j < numCities; j++) {
                ind.chromosome[j] = j;
            }
            // Shuffle the chromosome to create a random tour
            std::shuffle(ind.chromosome.begin(), ind.chromosome.end(), g);
            ind.fitness = 0.0f; // Initial fitness set to 0
            ind.islandID = island;
            popAssigned++;
        }
    }
}

// Compute the distance matrix between every pair of cities
void TSP::computeDistanceMatrix() {
    distanceMatrix.resize(numCities, std::vector<float>(numCities, 0.0f));
    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            distanceMatrix[i][j] = euclideanDistance(cities[i], cities[j]);
        }
    }
}

// Flatten the population into a one-dimensional array 'populationFlat'
// The order is: all individuals from island 0, then island 1, and so on.
void TSP::flattenPopulationToHost() {
    populationFlat.clear();
    populationFlat.reserve(popSize * numCities);

    // Iterate through each island
    for (int island = 0; island < numIslands; island++) {
        for (auto &ind : population[island]) {
            for (int c = 0; c < numCities; c++) {
                populationFlat.push_back(ind.chromosome[c]);
            }
        }
    }
}

// Flatten the distance matrix into a one-dimensional array 'distanceMatrixFlat' in row-major order.
void TSP::flattenDistanceMatrixToHost() {
    distanceMatrixFlat.clear();
    distanceMatrixFlat.reserve(numCities * numCities);
    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            distanceMatrixFlat.push_back(distanceMatrix[i][j]);
        }
    }
}

// Allocate GPU buffers for the flattened data and other required arrays.
void TSP::allocateGPUBuffers() {
    // Allocate GPU buffer for the flattened distance matrix
    cudaMalloc(&d_distanceMatrix, distanceMatrixFlat.size() * sizeof(float));
    // Allocate GPU buffer for the flattened population
    cudaMalloc(&d_population, populationFlat.size() * sizeof(int));
    cudaMalloc(&d_populationFitness, popSize * sizeof(float));
    // Allocate GPU buffers for parent A and parent B data
    cudaMalloc(&d_parentA, popSize * numCities * sizeof(int));
    cudaMalloc(&d_parentB, popSize * numCities * sizeof(int));
    cudaMalloc(&d_parentFitness, 2 * popSize * sizeof(float)); // Up to 2*popSize parent individuals
    // Allocate GPU buffers for children (resulting from crossover)
    cudaMalloc(&d_child1, popSize * numCities * sizeof(int));
    cudaMalloc(&d_child2, popSize * numCities * sizeof(int));
    // Allocate GPU buffers for offspring data
    cudaMalloc(&d_offspring, 2 * popSize * numCities * sizeof(int));
    cudaMalloc(&d_offspringFitness, 2 * popSize * sizeof(float));
    // Allocate GPU buffers for replacement operation
    cudaMalloc(&d_parentChromosomes, 2 * popSize * numCities * sizeof(int));
    cudaMalloc(&d_offspringChromosomes, 2 * popSize * numCities * sizeof(int));
}

// Copy the flattened population data from the host to the GPU.
void TSP::copyPopulationHostToDevice() {
    cudaMemcpy(d_population, populationFlat.data(),
               populationFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
}

// Copy the flattened distance matrix from the host to the GPU.
void TSP::copyDistanceMatrixHostToDevice() {
    cudaMemcpy(d_distanceMatrix, distanceMatrixFlat.data(),
               distanceMatrixFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
}

// Destructor: Release all allocated GPU buffers.
TSP::~TSP() {
    cudaFree(d_distanceMatrix);
    cudaFree(d_population);
    cudaFree(d_populationFitness);
    cudaFree(d_parentA);
    cudaFree(d_parentB);
    cudaFree(d_parentFitness);
    cudaFree(d_child1);
    cudaFree(d_child2);
    cudaFree(d_offspring);
    cudaFree(d_offspringFitness);
    cudaFree(d_parentChromosomes);
    cudaFree(d_offspringChromosomes);
}
