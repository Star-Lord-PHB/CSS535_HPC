#pragma once

#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>


/// @brief Helper data type representing a 2D Matrix
/// @tparam T type of the elements
/// @tparam Row row count
/// @tparam Col column count
template<typename T, typename = std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value>>
struct Matrix2D {

// private:
    std::vector<T> data;
    long _rowCount;

// public:
    inline long rowCount() const noexcept { return _rowCount; }
    inline long colCount() const noexcept { return data.size() / _rowCount; }
    inline long elementCount() const noexcept { return data.size(); }
    inline long byteSize() const noexcept { return elementCount() * sizeof(T); }
    inline const T* dataAs1DArray() const noexcept { return data.data(); }
    inline T* dataAs1DArray() noexcept { return &data[0]; }
    inline const T* operator[](long i) const noexcept { return &data[i * colCount()]; }
    inline T* operator[](long i) noexcept { return &data[i * colCount()]; }

    void randomInit() {
        for (auto i = 0; i < elementCount(); i++) {
            data[i] = static_cast<T>(rand()) / RAND_MAX;
        }
    }

    double diff(const Matrix2D<T>& other) const {
        if (other.rowCount() != rowCount() || other.colCount() != colCount()) {
            throw std::runtime_error("comparing matrix with different shape is not allowed");
        }
        double diffVal = 0;
        auto selfArr = dataAs1DArray();
        auto otherArr = other.dataAs1DArray();
        for (auto i = 0; i < other.rowCount() * other.colCount(); i++) {
            diffVal += std::pow(selfArr[i] - otherArr[i], 2);
        }
        return std::sqrt(diffVal);
    }

    Matrix2D(long rowCount, long colCount): _rowCount(rowCount), data(rowCount * colCount) {}
    // Matrix2D(const std::initializer_list<std::initializer_list<T>>& list) {
    //     if (list.size() > M) { throw std::length_error("Initializer list size exceeds Matrix row count"); }
    //     auto i = 0;
    //     for (const auto& row : list) {
    //         if (row.size() > N) { throw std::length_error("Initializer list size exceeds Matrix column count"); }
    //         std::copy(row.begin(), row.end(), data[i]);
    //         i++;
    //     }
    // }
    // Matrix2D(const std::initializer_list<T>& list) {
    //     if (list.size() > M * N) { throw std::length_error("Initializer list size exceeds Matrix size"); }
    //     T* ptr = &data[0][0];
    //     std::copy(list.begin(), list.end(), ptr);
    // }

    friend std::ostream& operator<<(std::ostream& os, const Matrix2D<T>& arr) {
        auto M = arr.rowCount();
        auto N = arr.colCount();
        os << "[";
        for (auto i = 0; i < M - 1; i++) {
            os << "[";
            for (auto j = 0; j < N - 1; j++) {
                os << arr[i][j] << " ";
            }
            os << arr[i][N - 1] << "] ";
        }
        os << "[";
        for (auto j = 0; j < N - 1; j++) {
            os << arr[M - 1][j] << " ";
        }
        os << arr[M - 1][N - 1] << "]]";
        return os;
    }
};