#ifndef TSP_H
#define TSP_H

#include <vector>

// 城市结构体
struct City {
    int id;
    float x;
    float y;
};

// 个体结构体（基于城市顺序的排列）
// 增加 islandID 用于标识所属岛屿
struct Individual {
    std::vector<int> chromosome; // 城市访问顺序
    float fitness;               // 适应度（路径总距离）
    int islandID;                // 所属岛的编号
};

class TSP {
public:
    int numCities;            // 城市数量
    int popSize;              // 整个种群大小（所有岛的总个体数）
    int mapSize;              // 地图范围
    int numIslands;           // 岛屿数量
    float parentSelectionRate;// 选为父代个体的百分比
    float crossoverProbability; // 交叉概率
    float mutationProbability;  // 变异概率

    std::vector<City> cities;
    // 种群被分割为多个岛，每个岛内为一子种群
    std::vector<std::vector<Individual>> population;
    // 采用二维 vector 表示距离矩阵，distanceMatrix[i][j] 表示城市 i 与城市 j 之间的距离
    std::vector<std::vector<float>> distanceMatrix;

    // 构造函数：新增参数传入岛屿数量、选择率、交叉概率和变异概率
    TSP(int _numCities, int _popSize, int _mapSize, int _numIslands,
        float _parentSelectionRate, float _crossoverProbability, float _mutationProbability);

    void initCities();
    void initPopulation();
    void computeDistanceMatrix();

    // 接口：将距离矩阵转换为 device 上的连续数组（注意：需要在 CUDA 内核中按照行主序访问）
    float* transferDistanceMatrixToDevice();
    // 接口：将种群转换为 device 上的连续整数数组，假设每个个体的染色体依次排列
    int* transferPopulationToDevice();
};

#endif
