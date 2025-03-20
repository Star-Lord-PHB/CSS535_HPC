#ifndef TSP_H
#define TSP_H

#include <vector>

// 城市结构体
struct City {
    int id;
    float x;
    float y;
};

// 个体结构体：基于城市顺序的排列
struct Individual {
    std::vector<int> chromosome;
    float fitness;
    int islandID; // 标记所属岛屿
};

// 在 TSP 中放入 ParentPairs 和 Offsprings 类型
// 保持与原来 GAInterface 中的别名一致
typedef std::vector<std::vector<std::pair<Individual, Individual>>> ParentPairs;
typedef std::vector<std::vector<Individual>> Offspring;

class TSP {
public:
    int numCities;             // 城市数量
    int popSize;               // 总种群大小（所有岛的个体总数）
    int mapSize;               // 地图尺寸（例如 100 表示 0~100 的范围）
    int numIslands;            // 岛屿数量
    float parentSelectionRate; // 父代选择百分比（可选参数）
    float crossoverProbability;// 交叉概率
    float mutationProbability; // 变异概率

    // ------------------
    // 原有数据结构
    // ------------------
    std::vector<City> cities;                         // 城市坐标
    std::vector<std::vector<Individual>> population;  // 按岛存储的种群
    std::vector<std::vector<float>> distanceMatrix;    // 二维距离矩阵

    // ------------------
    // 新增：GA相关中间结果
    // ------------------
    ParentPairs parentPairs;       // 按岛存储的父母对
    Offspring   offsprings;        // 按岛存储的后代

    // 为了方便 GPU/CPU 共享展平数据，额外存储以下：
    // 1) 种群展平数据
    std::vector<int> populationFlat;
    // 2) 距离矩阵展平数据 (row-major)
    std::vector<float> distanceMatrixFlat;
    // 3) 父母对（parentPairs）展平后 A、B 染色体
    std::vector<int> parentAFlat;
    std::vector<int> parentBFlat;
    std::vector<float> parentFitnessFlat; // 父代的适应度展平
    // 4) offspring（后代）展平后染色体及适应度
    std::vector<int> offspringFlat;
    std::vector<float> offspringFitnessFlat;

    // 每个岛上的配对数（用于 kernel 中确定并行大小）
    std::vector<int> parentPairCount;

    // ------------------
    // GPU 相关 device 端指针（如需持久化分配）
    // ------------------
    float *d_distanceMatrix = nullptr;
    int   *d_population     = nullptr;
    float *d_populationFitness = nullptr;

    // 父母对 / 子代 用于 CUDA kernel
    int   *d_parentA = nullptr;
    int   *d_parentB = nullptr;
    float *d_parentFitness = nullptr;
    int   *d_child1 = nullptr;
    int   *d_child2 = nullptr;

    // Offspring 用于 mutation/crossover
    int   *d_offspring = nullptr;
    float *d_offspringFitness = nullptr;

    // Replacement 用到的合并父母和子代
    int   *d_parentChromosomes = nullptr;
    int   *d_offspringChromosomes = nullptr;

    // 构造函数：传入城市数、种群大小、地图尺寸、岛屿数、选择率、交叉概率和变异概率
    TSP(int _numCities, int _popSize, int _mapSize, int _numIslands,
        float _parentSelectionRate, float _crossoverProbability, float _mutationProbability);

    // 初始化城市、种群、距离矩阵
    void initCities();
    void initPopulation();
    void computeDistanceMatrix();

    // 生成并拷贝到 GPU 的辅助函数
    void flattenPopulationToHost();
    void flattenDistanceMatrixToHost();
    void allocateGPUBuffers(); // 在此统一 cudaMalloc

    // 用于 GPU 的拷贝接口，如果要更新 GPU 的 population
    void copyPopulationHostToDevice();
    void copyDistanceMatrixHostToDevice();
};

#endif
