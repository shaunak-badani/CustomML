#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "read_data.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(CPPDataHandler, m) {
    m.doc() = "Module for reading data";
    
    py::class_<ReadData>(m, "ReadData")
    .def(py::init<MnistTrainType, MnistLabelsType>());
}