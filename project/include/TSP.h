#ifndef TSP_H
#define TSP_H

#include <vector>

// 城市结构体
struct City {
    int id;
    float x;
    float y;
};

// 个体结构体（基于城市顺序的排列，同时记录所属岛编号）
struct Individual {
    std::vector<int> chromosome;
    float fitness;
    int islandID;
};

class TSP {
public:
    int numCities;            // 城市数量
    int popSize;              // 总种群大小（所有岛的个体总数）
    int mapSize;              // 地图尺寸（例如 100 表示 0~100 的范围）
    int numIslands;           // 岛屿数量
    float parentSelectionRate;// 父代选择百分比（本示例中用于参数配置）
    float crossoverProbability; // 交叉概率
    float mutationProbability;  // 变异概率

    std::vector<City> cities;
    // 种群：每个岛一个子种群
    std::vector<std::vector<Individual>> population;
    // 距离矩阵：二维 vector 表示，distanceMatrix[i][j] 表示城市 i 与 j 间的距离
    std::vector<std::vector<float>> distanceMatrix;

    // 构造函数：传入城市数、种群大小、地图尺寸、岛屿数、选择率、交叉概率和变异概率
    TSP(int _numCities, int _popSize, int _mapSize, int _numIslands,
        float _parentSelectionRate, float _crossoverProbability, float _mutationProbability);

    void initCities();
    void initPopulation();
    void computeDistanceMatrix();

    // 将二维距离矩阵转换为连续数组，适合传到 GPU（CUDA 版本使用）
    float* transferDistanceMatrixToDevice();
    // 将种群转换为连续整数数组（平铺所有岛的个体），适合传到 GPU
    int* transferPopulationToDevice();
};

#endif
