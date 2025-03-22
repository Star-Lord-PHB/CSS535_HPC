#include <iostream>
#include "CpuGemv.hpp"
#include "Validate.hpp"
#include "Config.hpp"
#include <vector>
#include "NaiveGemv.hpp"
#include "GlobalMemory.hpp"



void runGlobalMemory() {

    const std::vector<Config> configs = {
        Config(
            KernelConfig(dim3(313, 1, 1), dim3(32, 1, 1)),
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(313, 1, 1), dim3(64, 1, 1)),
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(10, 1, 1), dim3(1024, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(100, 1, 1), dim3(1024, 1, 1)), 
            10000, 10000
        ),
        Config(
            KernelConfig(dim3(10, 1, 1), dim3(1000, 1, 1)), 
            10000, 10000
        ),
    };

    for (auto& config : configs) {

        auto mat = new Matrix2D<double>(config.M, config.N);
        auto vec = new Array<double>(config.N);
        auto vecOut = new Array<double>(config.M);

        mat->randomInit();
        vec->randomInit();
        vecOut->fill(0);

#if PRINTING
        std::cout << "mat: " << *mat << std::endl;
        std::cout << "vec: " << *vec << std::endl;
#endif 

#if CORRECTNESS_CHECK
        auto vecOutCPU = new Array<double>(config.M);
        cpu_gemv(*mat, *vec, *vecOutCPU);
#if PRINTING
        std::cout << "cpu: " << *vecOutCPU << std::endl;
#endif
#endif

        runNaiveOnce(config.kernelConfig, *mat, *vec, *vecOut);

#if CORRECTNESS_CHECK
        validate(*vecOut, *vecOutCPU);
#endif

        delete mat;
        delete vec;
        delete vecOut;
#if CORRECTNESS_CHECK
        delete vecOutCPU;
#endif 

    }

}