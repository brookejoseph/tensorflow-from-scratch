#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(minitf, m)
{
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<float> &, const std::vector<size_t> &>())
        .def("print", &Tensor::print);

    py::class_<Operation, AddOperation>(m, "AddOperation")
        .def(py::init<>())
        .def("forward", &AddOperation::forward);

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_operation", &Graph::add_operation)
        .def("run", &Graph::run);
}
