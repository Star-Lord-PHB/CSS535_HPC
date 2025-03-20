#ifndef GA_CPU_H
#define GA_CPU_H

#include "GAInterface.h"

namespace GA {
    ParentPairs selectionCPU(TSP &tsp);
    Offspring  crossoverCPU(TSP &tsp, const ParentPairs &parentPairs);
    void       mutationCPU(TSP &tsp, Offspring &offspring);
    void       replacementCPU(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring);
    void       migrationCPU(TSP &tsp);
    void       updatePopulationFitnessCPU(TSP &tsp);
    void       updateOffspringFitnessCPU(TSP &tsp, Offspring &offspring);
}

#endif
