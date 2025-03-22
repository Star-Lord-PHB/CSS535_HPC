#ifndef GA_INTERFACE_H
#define GA_INTERFACE_H

#include "TSP.h"

namespace GA {

    // The following interface functions operate solely on the TSP object.
    // They update tsp.parentPairs, tsp.offsprings, and other internal members as needed.
    void selection(TSP &tsp);
    void crossover(TSP &tsp);
    void mutation(TSP &tsp);
    void replacement(TSP &tsp);
    void migration(TSP &tsp);
    void updatePopulationFitness(TSP &tsp);
    void updateOffspringFitness(TSP &tsp);

} // namespace GA

#endif
