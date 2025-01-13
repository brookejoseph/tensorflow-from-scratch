#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>

class Tensor
{
public:
    std::vector<float> data;
    std::vector<size_t> shape;

    Tensor(const std::vector<float> &data, const std::vector<size_t> &shape);

    size_t size() const;

    void print() const;
};

#endif
