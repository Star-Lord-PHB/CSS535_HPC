//
// Created by admin on 18/03/2025.
//
// CUDA内核：计算适应度（路径距离）
__global__ void fitnessKernel(int* d_population, float* d_distanceMatrix,
                              float* d_fitness, int numCities, int popSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < popSize) {
        float totalDistance = 0.0f;
        for (int i = 0; i < numCities - 1; i++) {
            int city1 = d_population[idx * numCities + i];
            int city2 = d_population[idx * numCities + i + 1];
            totalDistance += d_distanceMatrix[city1 * numCities + city2];
        }
        // 回到起点
        int first = d_population[idx * numCities];
        int last = d_population[idx * numCities + numCities - 1];
        totalDistance += d_distanceMatrix[last * numCities + first];
        d_fitness[idx] = totalDistance;
    }
}