#pragma once
#include <functional>
#include <vector>
#include <stdexcept>
#include <ostream>


template<typename T>
struct Array2D {

private:
    const unsigned long _rowCount;
    std::vector<T> _data;

public:
    [[nodiscard]] inline size_t elementCount() const noexcept { return _data.size(); }
    [[nodiscard]] inline size_t byteCount() const noexcept { return elementCount() * sizeof(T); }
    [[nodiscard]] inline unsigned long rowCount() const noexcept { return _rowCount; }
    [[nodiscard]] inline unsigned long colCount() const noexcept { return _data.size() / _rowCount; }
    [[nodiscard]] inline const T* data() const { return _data.data(); }
    [[nodiscard]] inline T* data() { return _data.data(); }
    T* operator[] (unsigned long i) const {
        if (i >= _rowCount) {
            throw std::runtime_error("Array index out of bound");
        }
        return _data.data() + i * _rowCount;
    }
    const T& at(unsigned long i, unsigned long j) const {
        if (i >= _rowCount || j >= colCount()) {
            throw std::runtime_error("Array index out of bound");
        }
        return _data[i * colCount() + j];
    }
    T& at(unsigned long i, unsigned long j) {
        if (i >= _rowCount || j >= colCount()) {
            throw std::runtime_error("Array index out of bound");
        }
        return _data[i * colCount() + j];
    }

    void fill(std::function<T(unsigned long)> generator) noexcept {
        for (unsigned long i = 0; i < elementCount(); ++i) {
            _data[i] = generator(i);
        }
    }

    Array2D(unsigned long _rowCount, unsigned long _colCount) noexcept: _rowCount(_rowCount), _data(_rowCount * _colCount) {}
    Array2D(std::initializer_list<std::initializer_list<T>> initList): _rowCount(initList.size()) {
        if (initList.size() == 0) {
            _data = std::vector<T>();
            return;
        }
        auto _colCount = initList.begin()->size();
        if (_colCount == 0) {
            _data = std::vector<T>();
            return;
        }
        _data = std::vector<T>(_rowCount * _colCount);
        unsigned long i = 0;
        for (auto& row : initList) {
            std::copy(row.begin(), row.end(), _data.begin() + _colCount * i);
            i++;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Array2D<T>& a) {
        os << "[" << std::endl;
        for (unsigned long i = 0; i < a.rowCount(); i++) {
            os << "[";
            for (unsigned long j = 0; j < a.colCount(); j++) {
                os << a.at(i, j);
                if (j != a.colCount() - 1) { os << " "; }
            }
            os << "]" << std::endl;
        }
        os << "]";
        return os;
    }

};



template<typename T>
struct Array3D {

private:
    const unsigned long _xSize, _ySize;
    std::vector<T> _data;

public:
    [[nodiscard]] inline size_t elementCount() const noexcept { return _data.size(); }
    [[nodiscard]] inline size_t byteCount() const noexcept { return elementCount() * sizeof(T); }
    [[nodiscard]] inline unsigned long xSize() const noexcept { return _xSize; }
    [[nodiscard]] inline unsigned long ySize() const noexcept { return _ySize; }
    [[nodiscard]] inline unsigned long zSize() const noexcept { return _data.size() / xSize() / ySize(); }
    [[nodiscard]] inline const T* data() const { return _data.data(); }
    [[nodiscard]] inline T* data() { return _data.data(); }
    const T& at(unsigned long i, unsigned long j, unsigned long k) const {
        if (i >= _xSize || j >= ySize() || k >= zSize()) {
            throw std::runtime_error("Array index out of bound");
        }
        return _data[i * ySize() * zSize() + j * zSize() + k];
    }
    T& at(unsigned long i, unsigned long j, unsigned long k) {
        if (i >= _xSize || j >= ySize() || k >= zSize()) {
            throw std::runtime_error("Array index out of bound");
        }
        return _data[i * ySize() * zSize() + j * zSize() + k];
    }

    void fill(std::function<T(unsigned long)> generator) noexcept {
        for (unsigned long i = 0; i < elementCount(); ++i) {
            _data[i] = generator(i);
        }
    }

    void fill(std::function<T(unsigned long, unsigned long, unsigned long)> generator) noexcept {
        for (unsigned long i = 0; i < elementCount(); ++i) {
            _data[i] = generator(i / (ySize() * zSize()), i % (ySize() * zSize()) / zSize(), i % zSize());
        }
    }

    Array3D(unsigned long _xSize, unsigned long _ySize, unsigned long _zSize) noexcept: _xSize(_xSize), _ySize(_ySize), _data(_xSize * _ySize * _zSize) {}

    friend std::ostream& operator<<(std::ostream& os, const Array3D<T>& a) {
        os << "[" << std::endl;
        for (unsigned long i = 0; i < a.xSize(); i++) {
            os << "[";
            for (unsigned long j = 0; j < a.ySize(); j++) {
                os << "[";
                for (unsigned long k = 0; k < a.zSize(); k++) {
                    auto value = a.at(i, j, k);
                    // os << *reinterpret_cast<float*>(&value);
                    os << value;
                    if (k != a.zSize() - 1) { os << " "; }
                }
                os << "]";
            }
            os << "]" << std::endl;
        }
        os << "]";
        return os;
    }

};
