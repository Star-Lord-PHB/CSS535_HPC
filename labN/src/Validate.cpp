#include "Validate.hpp"


void validate(const Array<double>& vecOut, const Array<double>& vecOutCPU) {
    if (vecOutCPU.diff(vecOut) / vecOut.size() > 1e-10) {
        std::cerr << "GPU result is not the same as the CPU one" << std::endl;
        throw std::runtime_error("GPU result is not the same as the CPU one");
    }
}