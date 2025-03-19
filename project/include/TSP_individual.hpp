__global__ void fitnessKernel(int *d_population, float *d_distanceMatrix,
float *d_fitness, int numCities, int popSize);