#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#include <carma>

namespace py = pybind11;

#ifndef TESTS_SRC_DEBUG_H_
#define TESTS_SRC_DEBUG_H_

namespace carma {
namespace tests {

py::array_t<double> debug_mat_to_arr(py::array_t<double>& arr, bool copy);

py::array_t<double> debug_arr_to_mat(py::array_t<double>& arr, int copy);

}  // namespace tests
}  // namespace carma

void bind_debug_mat_to_arr(py::module& m);
void bind_debug_arr_to_mat(py::module& m);
#endif // TESTS_SRC_DEBUG_H_
