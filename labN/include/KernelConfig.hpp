#pragma once

#include <vector>
#include <vector_types.h>


struct KernelConfig {

    dim3 blockCount;
    dim3 threadCount;

    KernelConfig(dim3 blockCount, dim3 threadCount);

};