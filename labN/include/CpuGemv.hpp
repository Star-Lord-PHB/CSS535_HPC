#pragma once

#include "Array.hpp"
#include "Matrix.hpp"


void cpu_gemv(const Matrix2D<double>& mat, const Array<double>& vec, Array<double> &out);