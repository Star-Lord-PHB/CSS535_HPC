#pragma once

#include "Array.hpp"


void ga_one_thread_per_gene(
    Array3D<int>& solution,
    Array2D<float>& distanceMap
);


void ga_one_thread_per_gene_test();