#include "tsp.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// 计算欧几里得距离（内部使用）
static float euclideanDistance(const City& a, const City& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) +
                (a.y - b.y) * (a.y - b.y));
}

// 构造函数：传入城市数、种群总数、地图尺寸、岛屿数以及选择率、交叉概率和变异概率
TSP::TSP(int _numCities, int _popSize, int _mapSize, int _numIslands,
         float _parentSelectionRate, float _crossoverProbability, float _mutationProbability)
    : numCities(_numCities), popSize(_popSize), mapSize(_mapSize), numIslands(_numIslands),
      parentSelectionRate(_parentSelectionRate), crossoverProbability(_crossoverProbability),
      mutationProbability(_mutationProbability)
{
    srand(time(NULL));
    initCities();
    initPopulation();
    computeDistanceMatrix();
}

// 随机生成城市坐标
void TSP::initCities() {
    for (int i = 0; i < numCities; i++) {
        City c;
        c.id = i;
        c.x = static_cast<float>(rand() % mapSize);
        c.y = static_cast<float>(rand() % mapSize);
        cities.push_back(c);
    }
}

// 初始化种群：将总种群均分到各个岛中，并为每个个体生成随机的城市访问顺序
void TSP::initPopulation() {
    // 构造随机数引擎
    std::random_device rd;
    std::mt19937 g(rd());

    // 初始化种群二维 vector：每个岛作为一个子 vector
    population.resize(numIslands);
    int individualsPerIsland = popSize / numIslands;
    int remainder = popSize % numIslands; // 用于处理不能整除的情况

    for (int island = 0; island < numIslands; island++) {
        // 分配给当前岛的个体数量
        int currentIslandPop = individualsPerIsland + (island < remainder ? 1 : 0);
        for (int i = 0; i < currentIslandPop; i++) {
            Individual ind;
            ind.chromosome.resize(numCities);
            for (int j = 0; j < numCities; j++) {
                ind.chromosome[j] = j;
            }
            std::shuffle(ind.chromosome.begin(), ind.chromosome.end(), g);
            ind.fitness = 0.0f;      // 初始适应度可设为 0 或一个大值
            ind.islandID = island;   // 标记所属岛屿
            population[island].push_back(ind);
        }
    }
}

// 计算距离矩阵（使用二维 vector 表示，每个元素 distanceMatrix[i][j] 表示城市 i 与城市 j 的距离）
void TSP::computeDistanceMatrix() {
    distanceMatrix.resize(numCities);
    for (int i = 0; i < numCities; i++) {
        distanceMatrix[i].resize(numCities);
        for (int j = 0; j < numCities; j++) {
            distanceMatrix[i][j] = euclideanDistance(cities[i], cities[j]);
        }
    }
}

// 将二维距离矩阵转换为 device 上的连续数组（行主序排列）
float* TSP::transferDistanceMatrixToDevice() {
    float* d_distanceMatrix;
    cudaMalloc(&d_distanceMatrix, numCities * numCities * sizeof(float));

    // 将二维 distanceMatrix 平铺成一维数组
    std::vector<float> flat_distance;
    flat_distance.resize(numCities * numCities);
    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            flat_distance[i * numCities + j] = distanceMatrix[i][j];
        }
    }
    cudaMemcpy(d_distanceMatrix, flat_distance.data(),
               numCities * numCities * sizeof(float),
               cudaMemcpyHostToDevice);
    return d_distanceMatrix;
}

// 将二维种群（岛屿模式）转换为 device 上的连续整数数组
// 数组顺序为：先将所有岛的个体依次平铺，每个个体的染色体连续存储，长度均为 numCities
int* TSP::transferPopulationToDevice() {
    int totalGenes = 0;
    // 计算总基因数
    for (const auto& islandPop : population) {
        totalGenes += islandPop.size() * numCities;
    }
    std::vector<int> h_population;
    h_population.reserve(totalGenes);
    for (const auto& islandPop : population) {
        for (const auto& ind : islandPop) {
            for (int gene : ind.chromosome) {
                h_population.push_back(gene);
            }
        }
    }
    int* d_population;
    cudaMalloc(&d_population, totalGenes * sizeof(int));
    cudaMemcpy(d_population, h_population.data(),
               totalGenes * sizeof(int),
               cudaMemcpyHostToDevice);
    return d_population;
}
