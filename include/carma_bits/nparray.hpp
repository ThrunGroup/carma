/*  carma_bits/nparray.hpp: Condition checks numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */
#ifndef INCLUDE_CARMA_BITS_NPARRAY_HPP_
#define INCLUDE_CARMA_BITS_NPARRAY_HPP_

#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

#include <carma_bits/cnumpy.hpp> // NOLINT
#include <carma_bits/config.hpp> // NOLINT

#include <memory>
#include <type_traits>
#include <utility>

namespace py = pybind11;

namespace carma {

inline bool is_f_contiguous(const py::array& arr) {
    return details::is_f_contiguous(arr.ptr());
}

inline bool is_c_contiguous(const py::array& arr) {
    return details::is_c_contiguous(arr.ptr());
}

inline bool is_contiguous(const py::array& arr) {
    return is_f_contiguous(arr) || is_c_contiguous(arr);
}

inline bool is_writeable(const py::array& arr) {
    return details::is_writeable(arr.ptr());
}

inline bool is_owndata(const py::array& arr) {
    return details::is_owndata(arr.ptr());
}

inline bool is_aligned(const py::array& arr) {
    return  details::is_aligned(arr.ptr());
}

inline bool is_well_behaved(const py::array& arr) {
    return well_behaved(arr.ptr());
}

inline void set_owndata(py::array& arr) {
    return details::set_owndata(arr.ptr());
}

inline void set_not_owndata(py::array& arr) {
    return details::set_not_owndata(arr.ptr());
}

inline void set_writeable(py::array& arr) {
    return details::set_writeable(arr.ptr());
}

inline void set_not_writeable(py::array& arr) {
    return details::set_not_writeable(arr.ptr());
}

}  // namespace carma

#endif  // INCLUDE_CARMA_BITS_NPARRAY_HPP_
