#ifndef GA_CUDA_H
#define GA_CUDA_H

#include "GAInterface.h"

namespace GA {
    ParentPairs selectionCUDA(TSP &tsp);
    Offspring  crossoverCUDA(TSP &tsp, const ParentPairs &parentPairs);
    void       mutationCUDA(TSP &tsp, Offspring &offspring);
    void       replacementCUDA(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring);
    void       migrationCUDA(TSP &tsp);
    void       updatePopulationFitnessCUDA(TSP &tsp);
    void       updateOffspringFitnessCUDA(TSP &tsp, Offspring &offspring);
}

#endif
