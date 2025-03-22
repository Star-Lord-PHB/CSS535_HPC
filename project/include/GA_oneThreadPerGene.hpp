#pragma once

#include "Array.hpp"
#include <chrono>

using namespace std::chrono;


void ga_one_thread_per_gene(
    Array3D<int>& solution,
    const Array2D<float>& distanceMap,
    int generation,
    float crossoverProb,
    float mutationProb,
    milliseconds* calculationTime,
    milliseconds* totalTime
);


void ga_one_thread_per_gene_test();