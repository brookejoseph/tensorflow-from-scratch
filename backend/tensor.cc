#include "Tensor.h"

Tensor::Tensor(const std::vector<float> &data, const std::vector<size_t> &shape)
    : data(data), shape(shape) {}

size_t Tensor::size() const
{
    size_t s = 1;
    for (size_t dim : shape)
        s *= dim;
    return s;
}

void Tensor::print() const
{
    for (float val : data)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
