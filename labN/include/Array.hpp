#pragma once

#include <iostream>
#include <type_traits>
#include <initializer_list>
#include <vector>


/// @brief Helper data type representing an Array 
/// @tparam T type of the elements
/// @tparam N element count
template <typename T, typename = std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value>>
struct Array {

private:
    std::vector<T> _data;

public:
    inline T* data() noexcept { return _data.data(); }
    inline const T* data() const noexcept { return _data.data(); }
    inline long size() const noexcept { return _data.size(); }
    inline long byteSize() const noexcept { return size() * sizeof(T); }
    inline T& operator[](long i) noexcept { return _data[i]; }
    inline const T& operator[](long i) const noexcept { return _data[i]; }

    void randomInit() {
        for (auto i = 0; i < size(); i++) {
            _data[i] = static_cast<T>(rand()) / RAND_MAX;
        }
    }

    void fill(T value) {
        std::fill(_data.begin(), _data.end(), value);
    }

    double diff(const Array<T>& other) const {
        double diff = 0;
        for (auto i = 0; i < other.size(); i++) {
            diff += std::pow(_data[i] - other[i], 2);
        }
        return std::sqrt(diff);
    }

    Array(long size): _data(size) {} 
    Array(const std::initializer_list<T>& list) {
        std::copy(list.begin(), list.end(), _data);
    }

    bool operator==(const Array<T>& other) const {
        for (auto i = 0; i < size(); i++) {
            if (_data[i] != other[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const Array<T>& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os, const Array<T>& arr) {
        auto N = arr.size();
        os << "[";
        for (auto i = 0; i < N - 1; i++) {
            os << arr[i] << " ";
        }
        os << arr[N - 1] << "]";
        return os;
    }

};