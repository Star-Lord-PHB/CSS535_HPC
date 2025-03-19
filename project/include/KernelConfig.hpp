#pragma once

#include <vector>
#include <vector_types.h>


/// @brief Kernel execution configuration, including the grid size and block size
struct KernelConfig {

    dim3 blockCount;
    dim3 threadCount;

    KernelConfig(dim3 blockCount, dim3 threadCount);

};