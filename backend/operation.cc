#include "Operation.h"
#include <stdexcept>

Tensor AddOperation::forward(const Tensor &input1, const Tensor &input2)
{
    if (input1.shape != input2.shape)
    {
        throw std::invalid_argument("Input tensors must have the same shape for addition.");
    }

    std::vector<float> result_data;
    for (size_t i = 0; i < input1.data.size(); ++i)
    {
        result_data.push_back(input1.data[i] + input2.data[i]);
    }
    return Tensor(result_data, input1.shape);
}
