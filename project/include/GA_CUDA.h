#ifndef GA_CUDA_H
#define GA_CUDA_H

#include "GAInterface.h"

namespace GA {
    void selectionCUDA(TSP &tsp);
    void crossoverCUDA(TSP &tsp);
    void mutationCUDA(TSP &tsp);
    void replacementCUDA(TSP &tsp);
    void migrationCUDA(TSP &tsp);
    void updatePopulationFitnessCUDA(TSP &tsp);
    void updateOffspringFitnessCUDA(TSP &tsp);

}

#endif
