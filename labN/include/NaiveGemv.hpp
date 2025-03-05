#pragma once

#include "Array.hpp"
#include "KernelConfig.hpp"
#include "Matrix.hpp"


void runNaiveOnce(
    const KernelConfig kernelConfig, 
    const Matrix2D<double>& mat,
    const Array<double>& vec,
    Array<double>& vecOut
);
void runNaive();