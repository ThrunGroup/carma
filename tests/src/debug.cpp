#include "debug.h"

namespace carma {
namespace tests {

py::array_t<double> debug_mat_to_arr(py::array_t<double>& arr, bool copy) {
    arma::mat mat = arma::mat(4, 5, arma::fill::randu);
    return mat_to_arr(mat, copy);
}

py::array_t<double> debug_arr_to_mat(py::array_t<double>& arr, int copy) {
    if (copy < 0) {
        return carma::mat_to_arr(carma::arr_to_mat<double>(std::move(arr)));
    }
    return carma::mat_to_arr(carma::arr_to_mat<double>(arr, copy));
}


}  // namespace tests
}  // namespace carma

void bind_debug_mat_to_arr(py::module& m) {
    m.def("debug_mat_to_arr", &carma::tests::debug_mat_to_arr, "Test arr_to_mat_double");
}

void bind_debug_arr_to_mat(py::module& m) {
    m.def("debug_arr_to_mat", &carma::tests::debug_arr_to_mat, "Test arr_to_mat_double");
}

