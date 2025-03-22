#pragma once

#include "KernelConfig.hpp"

struct Config {
    KernelConfig kernelConfig;
    long M;
    long N;
    Config(KernelConfig kernelConfig, long M, long N);
};