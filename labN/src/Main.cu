#include <iostream>
#include "Array.hpp"
#include "Matrix.hpp"
#include "CpuGemv.hpp"
#include <vector>
#include <chrono>

using namespace std::chrono;


#ifndef CORRECTNESS_CHECK
#define CORRECTNESS_CHECK false
#endif

#ifndef JSON
#define JSON false  // whether to print in json format
#endif 

#ifndef PRINTING
#define PRINTING false  // whether to print detailed input and output of each run
#endif 

#define NUM_RUN 10


#include "ConfigSet.hpp"


int main(void) {

    // runNaive();
    // runGlobalMemory();
    // runShareMemory();
    runRegisters();

}