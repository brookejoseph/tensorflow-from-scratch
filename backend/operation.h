#ifndef OPERATION_H
#define OPERATION_H

#include "Tensor.h"

class Operation
{
public:
    virtual ~Operation() = default;

    virtual Tensor forward(const Tensor &input1, const Tensor &input2) = 0;
};

class AddOperation : public Operation
{
public:
    Tensor forward(const Tensor &input1, const Tensor &input2) override;
};

#endif
