#include "CpuGemv.hpp"



void cpu_gemv(const Matrix2D<double> &mat, const Array<double> &vec, Array<double> &out) {
    
    for (auto i = 0; i < mat.rowCount(); i++) {
        for (auto j = 0; j < mat.colCount(); j++) {
            out[i] += mat[i][j] * vec[j];
        }
    }

}
