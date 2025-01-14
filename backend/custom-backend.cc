#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;

REGISTER_OP("CustomQuantize")
    .Input("input: float")
    .Output("output: float")
    .Attr("bit_width: int")
    .SetShapeFn([](shape_inference::InferenceContext *c)
                {
        c->set_output(0, c->input(0));
        return Status::OK(); });

class CustomQuantizeOp : public OpKernel
{
public:
    explicit CustomQuantizeOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("bit_width", &bit_width_));
        OP_REQUIRES(context, bit_width_ > 0, errors::InvalidArgument("bit_width must be positive"));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input_tensor = context->input(0);
        auto input_flat = input_tensor.flat<float>();

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        const float scale = (1 << bit_width_) - 1;
        for (int i = 0; i < input_flat.size(); ++i)
        {
            float scaled_value = input_flat(i) * scale;
            output_flat(i) = std::round(scaled_value) / scale;
        }
    }

private:
    int bit_width_;
};

REGISTER_KERNEL_BUILDER(Name("CustomQuantize").Device(DEVICE_CPU), CustomQuantizeOp);

PYBIND11_MODULE(custom_backend, m)
{
    m.doc() = "Custom backend for TensorFlow quantization operations";
}
