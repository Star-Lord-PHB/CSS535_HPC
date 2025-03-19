#include "KernelConfig.hpp"


KernelConfig::KernelConfig(dim3 blockCount, dim3 threadCount): blockCount(blockCount), threadCount(threadCount) {}
