/*  carma_bits/armatonumpy.hpp: Coverter of Armadillo matrices to numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */
#ifndef INCLUDE_CARMA_BITS_ARMATONUMPY_HPP_
#define INCLUDE_CARMA_BITS_ARMATONUMPY_HPP_
#include <pybind11/pybind11.h>  // NOLINT
#include <pybind11/numpy.h>  // NOLINT
#include <carma_bits/config.hpp> // NOLINT

#include <armadillo>  // NOLINT
#include <utility>

namespace py = pybind11;

namespace carma {
namespace details {

template <typename armaT>
inline py::capsule create_capsule(armaT* data) {
    return py::capsule(data, [](void* f) {
        armaT* mat = reinterpret_cast<armaT*>(f);
#ifdef CARMA_EXTRA_DEBUG
        std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
        // if in debug mode let us know what pointer is being freed
        std::cerr << "Freeing memory @" << mat->memptr() << std::endl;
        std::cout << "-----------" << "\n";
#endif
        delete mat;
    });
} /* create_capsule */

template <typename T>
inline py::capsule create_dummy_capsule(T* data) {
    return py::capsule(data, [](void* f) {
#ifdef CARMA_EXTRA_DEBUG
        std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
        // if in debug mode let us know what pointer is being freed
        std::cerr << "Destructing view on memory @" << f << std::endl;
        std::cout << "-----------" << "\n";
#endif
    });
} /* create_capsule */


template <typename T>
inline py::array_t<T> construct_array(arma::Row<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);

    py::capsule base = create_capsule<arma::Row<T>>(data);

#ifdef CARMA_C_CONTIGUOUS_MODE
    return py::array_t<T>(
        {static_cast<ssize_t>(1), ncols},  // shape
        {ncols * tsize, tsize},            // C-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
#else
    return py::array_t<T>(
        {static_cast<ssize_t>(1), ncols},  // shape
        {tsize, tsize},                    // F-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
#endif
} /* construct_array */

template <typename T>
inline py::array_t<T> construct_array(arma::Col<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);

    py::capsule base = create_capsule<arma::Col<T>>(data);

#ifdef CARMA_C_CONTIGUOUS_MODE
    return py::array_t<T>(
        {nrows, static_cast<ssize_t>(1)},  // shape
        {tsize, tsize},                    // C-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
#else
    return py::array_t<T>(
        {nrows, static_cast<ssize_t>(1)},  // shape
        {tsize, nrows * tsize},            // F-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
#endif
} /* construct_array */

template <typename T>
inline py::array_t<T> construct_array(arma::Mat<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);

    py::capsule base = create_capsule<arma::Mat<T>>(data);

#ifdef CARMA_C_CONTIGUOUS_MODE
    return py::array_t<T>(
        {nrows, ncols},          // shape
        {ncols * tsize, tsize},  // C-style contiguous strides
        data->memptr(),          // the data pointer
        base                     // numpy array references this parent
    );
#else
    return py::array_t<T>(
        {nrows, ncols},          // shape
        {tsize, nrows * tsize},  // F-style contiguous strides
        data->memptr(),          // the data pointer
        base                     // numpy array references this parent
    );
#endif
} /* construct_array */

template <typename T>
inline py::array_t<T> construct_array(arma::Cube<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);
    ssize_t nslices = static_cast<ssize_t>(data->n_slices);

#ifdef CARMA_C_CONTIGUOUS_MODE
    return py::array_t<T>(
        // shape
        {nrows, ncols, nslices},
        // C-style contiguous strides
        {ncols * nslices * tsize, nslices * tsize, tsize},
        // the data pointer
        data->memptr(),
        // numpy array references this parent
        create_capsule<arma::Cube<T>>(data)
    );
#else
    return py::array_t<T>(
        // shape
        {nrows, ncols, nslices},
        // F-style contiguous strides
        {tsize, nrows * tsize, tsize * nrows * ncols},
        // the data pointer
        data->memptr(),
        // numpy array references this parent
        create_capsule<arma::Cube<T>>(data)
    );
#endif
} /* construct_array */

}  // namespace details
}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_ARMATONUMPY_HPP_
