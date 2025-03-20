#ifndef GA_INTERFACE_H
#define GA_INTERFACE_H

#include "TSP.h"
#include <vector>
#include <utility>

namespace GA {

    ParentPairs selection(TSP &tsp);
    Offspring  crossover(TSP &tsp, const ParentPairs &parentPairs);
    void       mutation(TSP &tsp, Offspring &offspring);
    void       replacement(TSP &tsp, const ParentPairs &parentPairs, const Offspring &offspring);
    void       migration(TSP &tsp);
    void       updatePopulationFitness(TSP &tsp);
    void       updateOffspringFitness(TSP &tsp, Offspring &offspring);

} // namespace GA

#endif
