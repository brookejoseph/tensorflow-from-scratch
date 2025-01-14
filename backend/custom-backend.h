#ifndef CUSTOM_BACKEND_H
#define CUSTOM_BACKEND_H

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace tensorflow
{

    class CustomQuantizeOp : public OpKernel
    {
    public:
        explicit CustomQuantizeOp(OpKernelConstruction *context);

        void Compute(OpKernelContext *context) override;

    private:
        int bit_width_;
    };

}

PYBIND11_MODULE(custom_backend, m);

#endif
