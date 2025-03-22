#ifndef GA_CPU_H
#define GA_CPU_H

#include "GAInterface.h"

namespace GA {
    void selectionCPU(TSP &tsp);
    void crossoverCPU(TSP &tsp);
    void mutationCPU(TSP &tsp);
    void replacementCPU(TSP &tsp);
    void migrationCPU(TSP &tsp);
    void updatePopulationFitnessCPU(TSP &tsp);
    void updateOffspringFitnessCPU(TSP &tsp);

}

#endif
