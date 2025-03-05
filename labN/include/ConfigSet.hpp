#pragma once

#include "NaiveGemv.hpp"
#include "GlobalMemory.hpp"
#include "ShareMemory.hpp"
#include "Registers.hpp"


struct Config {
    KernelConfig kernelConfig;
    long M;
    long N;
    Config(KernelConfig kernelConfig, long M, long N);
};