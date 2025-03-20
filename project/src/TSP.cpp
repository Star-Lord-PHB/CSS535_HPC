#include "TSP.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// 计算欧几里得距离（内部使用）
static float euclideanDistance(const City &a, const City &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 构造函数
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
    // 1) 初始化城市与种群
    initCities();
    initPopulation();
    // 2) 计算距离矩阵
    computeDistanceMatrix();
    // 3) 在 Host 上展平种群与距离矩阵
    flattenPopulationToHost();
    flattenDistanceMatrixToHost();
    // 4) 分配 GPU 缓冲区
    allocateGPUBuffers();
    // 并将数据复制到 GPU
    copyPopulationHostToDevice();
    copyDistanceMatrixHostToDevice();
}

// 随机生成城市坐标
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

// 初始化种群：将总种群均分到各个岛中，随机打乱染色体
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
            for (int j = 0; j < numCities; j++) {
                ind.chromosome[j] = j;
            }
            std::shuffle(ind.chromosome.begin(), ind.chromosome.end(), g);
            ind.fitness = 0.0f;
            ind.islandID = island;
            popAssigned++;
        }
    }
}

// 计算距离矩阵
void TSP::computeDistanceMatrix() {
    distanceMatrix.resize(numCities, std::vector<float>(numCities, 0.0f));
    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            distanceMatrix[i][j] = euclideanDistance(cities[i], cities[j]);
        }
    }
}

// 将 population 展平到 populationFlat
void TSP::flattenPopulationToHost() {
    populationFlat.clear();
    populationFlat.reserve(popSize * numCities);

    // 依次遍历每个岛
    for (int island = 0; island < numIslands; island++) {
        for (auto &ind : population[island]) {
            for (int c = 0; c < numCities; c++) {
                populationFlat.push_back(ind.chromosome[c]);
            }
        }
    }
}

// 将 distanceMatrix 展平到 distanceMatrixFlat (row-major)
void TSP::flattenDistanceMatrixToHost() {
    distanceMatrixFlat.clear();
    distanceMatrixFlat.reserve(numCities * numCities);
    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            distanceMatrixFlat.push_back(distanceMatrix[i][j]);
        }
    }
}

// 统一分配 GPU 缓冲区
void TSP::allocateGPUBuffers() {
    // 距离矩阵
    cudaMalloc(&d_distanceMatrix, distanceMatrixFlat.size() * sizeof(float));
    // 种群
    cudaMalloc(&d_population, populationFlat.size() * sizeof(int));
    cudaMalloc(&d_populationFitness, popSize * sizeof(float));
    // parent A / B
    cudaMalloc(&d_parentA, popSize * numCities * sizeof(int));
    cudaMalloc(&d_parentB, popSize * numCities * sizeof(int));
    cudaMalloc(&d_parentFitness, 2 * popSize * sizeof(float)); // 最多2*popSize个父代
    // child buffer
    cudaMalloc(&d_child1, popSize * numCities * sizeof(int));
    cudaMalloc(&d_child2, popSize * numCities * sizeof(int));
    // offspring
    cudaMalloc(&d_offspring, 2 * popSize * numCities * sizeof(int));
    cudaMalloc(&d_offspringFitness, 2 * popSize * sizeof(float));
    // replacement
    cudaMalloc(&d_parentChromosomes, 2 * popSize * numCities * sizeof(int));
    cudaMalloc(&d_offspringChromosomes, 2 * popSize * numCities * sizeof(int));
}

// 将 Host 上的 populationFlat 拷贝到 GPU
void TSP::copyPopulationHostToDevice() {
    cudaMemcpy(d_population, populationFlat.data(),
               populationFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
}

// 将 Host 上的 distanceMatrixFlat 拷贝到 GPU
void TSP::copyDistanceMatrixHostToDevice() {
    cudaMemcpy(d_distanceMatrix, distanceMatrixFlat.data(),
               distanceMatrixFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
}
