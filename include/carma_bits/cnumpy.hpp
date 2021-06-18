/*  carma_bits/cnumpy.h: Code to steal the memory from Numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */

#ifndef INCLUDE_CARMA_BITS_CNUMPY_HPP_
#define INCLUDE_CARMA_BITS_CNUMPY_HPP_
#include <object.h>
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include <Python.h>
#include <pymem.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <carma_bits/config.hpp>
#include <carma_bits/numpyapi.hpp>
#include <carma_bits/debug.hpp>
#include <carma_bits/typecheck.hpp>

#include <armadillo>

#include <limits>
#include <iostream>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <cstdint>

namespace py = pybind11;

extern "C" {
#if defined CARMA_C_CONTIGUOUS_MODE
    /* well behaved is defined as:
     *   - aligned
     *   - writeable
     *   - C contiguous (row major)
     *   - owndata (optional, on by default)
     * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
     */
    static inline bool well_behaved(PyObject* src) {
        PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(src);
    #if defined CARMA_DONT_REQUIRE_OWNDATA
        return PyArray_CHKFLAGS(
            arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS
        );
    #else
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_OWNDATA
        );
    #endif
    }
#else
    /* well behaved is defined as:
     *   - aligned
     *   - writeable
     *   - Fortran contiguous aka column major (optional, on by default)
     *   - owndata (optional, on by default)
     * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
     */
    static inline bool well_behaved(PyObject* src) {
        PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(src);
    #if defined CARMA_DONT_REQUIRE_OWNDATA && defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
        return PyArray_CHKFLAGS(
            arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
        );
    #elif defined CARMA_DONT_REQUIRE_OWNDATA
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS
        );
    #elif defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA
        );
    #else
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA
        );
    #endif
    }
#endif

#if defined CARMA_C_CONTIGUOUS_MODE
    /* well behaved is defined as:
     *   - aligned
     *   - C contiguous (row major)
     *   - owndata (optional, on by default)
     * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
     */
    static inline bool well_behaved_view(PyObject* src) {
        PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(src);
    #if defined CARMA_DONT_REQUIRE_OWNDATA
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS
        );
    #else
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_OWNDATA
        );
    #endif
    }
#else
    /* well behaved is defined as:
     *   - aligned
     *   - Fortran contiguous aka column major (optional, on by default)
     *   - owndata (optional, on by default)
     * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
     */
    static inline bool well_behaved_view(PyObject* src) {
        PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(src);
    #if defined CARMA_DONT_REQUIRE_OWNDATA && defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
        return PyArray_CHKFLAGS(
            arr, NPY_ARRAY_ALIGNED
        );
    #elif defined CARMA_DONT_REQUIRE_OWNDATA
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS
        );
    #elif defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_OWNDATA
        );
    #else
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA
        );
    #endif
    }
#endif

#if defined CARMA_C_CONTIGUOUS_MODE
    /* well behaved is defined as:
     *   - aligned
     *   - writeable
     *   - C contiguous (row major)
     *   - owndata (optional, on by default)
     * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
     */
    static inline bool well_behaved_arr(PyArrayObject* arr) {
    #if defined CARMA_DONT_REQUIRE_OWNDATA
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS
        );
    #else
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_OWNDATA
        );
    #endif
    }
#else
    /* well behaved is defined as:
     *   - aligned
     *   - writeable
     *   - Fortran contiguous aka column major (optional, on by default)
     *   - owndata (optional, on by default)
     * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
     */
    static inline bool well_behaved_arr(PyArrayObject* arr) {
    #if defined CARMA_DONT_REQUIRE_OWNDATA && defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
        return PyArray_CHKFLAGS(
            arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
        );
    #elif defined CARMA_DONT_REQUIRE_OWNDATA
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS
        );
    #elif defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA
        );
    #else
        return PyArray_CHKFLAGS(
            arr,
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA
        );
    #endif
    }
#endif
}  // extern "C"

namespace carma {
namespace details {

struct not_writeable_error : std::exception {
    const char* message;
    explicit not_writeable_error(const char* message) : message(message) {}
    const char* what() const throw() { return message; }
};

inline bool is_f_contiguous(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_f_contiguous(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_c_contiguous(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_c_contiguous(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_owndata(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src,  NPY_ARRAY_OWNDATA);
}

inline bool is_owndata(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    return PyArray_CHKFLAGS(src,  NPY_ARRAY_OWNDATA);
}

inline bool is_writeable(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src,  NPY_ARRAY_WRITEABLE);
}

inline bool is_writeable(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    return PyArray_CHKFLAGS(src,  NPY_ARRAY_WRITEABLE);
}

inline bool is_aligned(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src,  NPY_ARRAY_ALIGNED);
}

inline bool is_aligned(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    return PyArray_CHKFLAGS(src,  NPY_ARRAY_ALIGNED);
}

inline void set_owndata(PyArrayObject* src) {
    PyArray_ENABLEFLAGS(src,  NPY_ARRAY_OWNDATA);
}

inline void set_owndata(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    PyArray_ENABLEFLAGS(src,  NPY_ARRAY_OWNDATA);
}

inline void set_not_owndata(PyArrayObject* src) {
    PyArray_CLEARFLAGS(src,  NPY_ARRAY_OWNDATA);
}

inline void set_not_owndata(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    PyArray_CLEARFLAGS(src,  NPY_ARRAY_OWNDATA);
}

inline void set_writeable(PyArrayObject* src) {
    PyArray_ENABLEFLAGS(src,  NPY_ARRAY_WRITEABLE);
}

inline void set_writeable(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    PyArray_ENABLEFLAGS(src,  NPY_ARRAY_WRITEABLE);
}

inline void set_not_writeable(PyArrayObject* src) {
    PyArray_CLEARFLAGS(src,  NPY_ARRAY_WRITEABLE);
}

inline void set_not_writeable(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    PyArray_CLEARFLAGS(src,  NPY_ARRAY_WRITEABLE);
}

// grab corresponding NPY_TYPE at compile time
template<typename T>
constexpr int get_type() {
    bool pass_assert = true;
    if (std::is_same<bool, T>::value) {
        return NPY_BOOL;
    } else if (std::is_same<std::int8_t, T>::value) {
        return NPY_INT8;
    } else if (std::is_same<std::int16_t, T>::value) {
        return NPY_INT16;
    } else if (std::is_same<std::int32_t, T>::value) {
        return NPY_INT32;
    } else if (std::is_same<std::int64_t, T>::value) {
        return NPY_INT64;
    } else if (std::is_same<long, T>::value) {
        return NPY_LONG;
    } else if (std::is_same<size_t, T>::value) {
        return NPY_INTP;
    } else if (std::is_same<std::uint8_t, T>::value) {
        return NPY_UINT8;
    } else if (std::is_same<std::uint16_t, T>::value) {
        return NPY_UINT16;
    } else if (std::is_same<std::uint32_t, T>::value) {
        return NPY_UINT32;
    } else if (std::is_same<std::uint64_t, T>::value) {
        return NPY_UINT64;
    } else if (std::is_same<float, T>::value) {
        return NPY_FLOAT;
    } else if (std::is_same<double, T>::value) {
        return NPY_DOUBLE;
    } else if (std::is_same<std::complex<float>, T>::value) {
        return NPY_COMPLEX64;
    } else if (std::is_same<std::complex<double>, T>::value) {
        return NPY_COMPLEX128;
    } else {
        return -1;
    }
}

/* ---- steal_memory ----
 * The default behaviour is to turn off the owndata flag, numpy will no longer
 * free the allocated resources.
 * Benefit of this approach is that it's doesn't rely on deprecated access.
 * However, it can result in hard to detect bugs
 *
 * If CARMA_SOFT_STEAL is defined, the stolen array is replaced with an array
 * containing a single NaN and set the appropriate dimensions and strides.
 * This means the original references can be accessed but no longer should.
 *
 * Alternative is to define CARMA_HARD_STEAL which sets a nullptr and decreases
 * the reference count. NOTE, accessing the original reference when using
 * CARMA_HARD_STEAL will trigger a segfault.
 *
 * Note this function makes use of PyArrayObject_fields which is internal
 * and is noted with:
 *
 * "The main array object structure. It has been recommended to use the inline
 * functions defined below (PyArray_DATA and friends) to access fields here
 * for a number of releases. Direct access to the members themselves is
 * deprecated. To ensure that your code does not use deprecated access,
 * #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION (or
 * NPY_1_8_API_VERSION or higher as required).
 * This struct will be moved to a private header in a future release"
 */
template <typename T>
static inline void steal_memory(PyObject* src) {
#ifdef CARMA_EXTRA_DEBUG
    PyArrayObject* db_arr = reinterpret_cast<PyArrayObject*>(src);
    std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
    T* db_data = reinterpret_cast<T*>(PyArray_DATA(db_arr));
    std::cout << "Array with data adress: " << db_data << " will be stolen." << "\n";
    debug::print_array_info<T>(src);
    std::cout << "-----------" << "\n";
#endif
#if defined CARMA_HARD_STEAL
    reinterpret_cast<PyArrayObject_fields *>(src)->data = nullptr;
#elif defined CARMA_SOFT_STEAL
    PyArrayObject_fields* obj = reinterpret_cast<PyArrayObject_fields *>(src);
    double* data = reinterpret_cast<double *>(
            carman::npy_api::get().PyDataMem_NEW_(sizeof(double))
    );
    if (data == NULL) throw std::bad_alloc();
    data[0] = NAN;
    obj->data = reinterpret_cast<char*>(data);

    size_t ndim = obj->nd;
    obj->nd = 1;
    if (ndim == 1) {
        obj->dimensions[0] = static_cast<npy_int>(1);
    } else if (ndim == 2) {
        obj->dimensions[0] = static_cast<npy_int>(1);
        obj->dimensions[1] = static_cast<npy_int>(0);
    } else {
        obj->dimensions[0] = static_cast<npy_int>(1);
        obj->dimensions[1] = static_cast<npy_int>(0);
        obj->dimensions[2] = static_cast<npy_int>(0);
    }
#else
    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(src), NPY_ARRAY_OWNDATA);
#endif
}  // steal_memory

/* Use Numpy's api to account for stride, order and steal the memory */
template <typename T>
inline static T* steal_copy_array(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
#ifdef CARMA_EXTRA_DEBUG
    std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
    T* db_data = reinterpret_cast<T*>(PyArray_DATA(src));
    std::cout << "A copy of array with data adress @" << db_data << " will be stolen\n";
    debug::print_array_info<T>(obj);
    std::cout << "-----------" << "\n";
#endif
    PyArray_Descr* dtype = PyArray_DESCR(src);
    // NewFromDescr steals a reference
    Py_INCREF(dtype);
    // dimension checks have been done prior so array should
    // not have more than 3 dimensions
    int ndim = PyArray_NDIM(src);
    npy_intp const* dims = PyArray_DIMS(src);

    auto& api = carman::npy_api::get();
    // data will be freed by arma::memory::release
    T* data = arma::memory::acquire<T>(api.PyArray_Size_(obj));

    // build an PyArray to do F-order copy
    auto dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        dtype,
        ndim,
        dims,
        NULL,
        data,
#ifdef CARMA_C_CONTIGUOUS_MODE
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#else
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#endif
        NULL
    ));

    // copy the array to a well behaved F-order
    api.PyArray_CopyInto_(dest, src);

    // set OWNDATA to false such that the newly create
    // memory is not freed when the array is cleared
    PyArray_CLEARFLAGS(dest, NPY_ARRAY_OWNDATA);
    // free the array but not the memory
    api.PyArray_Free_(dest, static_cast<void*>(nullptr));
    return data;
}  // steal_copy_array

/* Use Numpy's api to account for stride, order and steal the memory */
template <typename T>
inline static T* swap_copy_array(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
#ifdef CARMA_EXTRA_DEBUG
    std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
    T* db_data = reinterpret_cast<T*>(PyArray_DATA(src));
    std::cout << "A copy of array with data adress @" << db_data << " will be swapped in place\n";
    debug::print_array_info<T>(obj);
    std::cout << "-----------" << "\n";
#endif
    if (!PyArray_CHKFLAGS(src, NPY_ARRAY_WRITEABLE)) {
        throw not_writeable_error("carma: Array is not writeable and cannot be swapped");
    }
    PyArray_Descr* dtype = PyArray_DESCR(src);
    // NewFromDescr steals a reference
    Py_INCREF(dtype);
    // dimension checks have been done prior so array should
    // not have more than 3 dimensions
    int ndim = PyArray_NDIM(src);
    npy_intp const* dims = PyArray_DIMS(src);

    auto& api = carman::npy_api::get();

    // build an PyArray to do F-order copy, memory will be freed by arma::memory::release
    auto tmp = reinterpret_cast<PyArrayObject*>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        dtype,
        ndim,
        dims,
        NULL,
        arma::memory::acquire<T>(api.PyArray_Size_(obj)),
#ifdef CARMA_C_CONTIGUOUS_MODE
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#else
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#endif
        NULL
    ));

    // copy the array to a well behaved F-order
    int ret_code = api.PyArray_CopyInto_(tmp, src);
    // swap copy into the original array
    auto tmp_of = reinterpret_cast<PyArrayObject_fields *>(tmp);
    auto src_of = reinterpret_cast<PyArrayObject_fields *>(src);
    std::swap(src_of->data, tmp_of->data);
    std::swap(src_of->strides, tmp_of->strides);

    if (PyArray_CHKFLAGS(src, NPY_ARRAY_OWNDATA)) {
        PyArray_ENABLEFLAGS(tmp, NPY_ARRAY_OWNDATA);
    }
#ifdef CARMA_C_CONTIGUOUS_MODE
    PyArray_ENABLEFLAGS(src, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED | NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS(src, NPY_ARRAY_F_CONTIGUOUS);
#else
    PyArray_ENABLEFLAGS(src, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED | NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS(src, NPY_ARRAY_C_CONTIGUOUS);
#endif

    Py_DECREF(tmp);
    return reinterpret_cast<T*>(PyArray_DATA(src));
}  // swap_copy_array

template <typename T>
inline static PyObject* copy_array(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    PyArray_Descr* dtype = PyArray_DESCR(src);
    // NewFromDescr steals a reference
    Py_INCREF(dtype);
    // dimension checks have been done prior so array should
    // not have more than 3 dimensions
    int ndim = PyArray_NDIM(src);
    npy_intp const* dims = PyArray_DIMS(src);

    auto& api = carman::npy_api::get();
    // data will be freed by arma::memory::release
    T* data = arma::memory::acquire<T>(api.PyArray_Size_(obj));

    // build an PyArray to do F-order copy
    auto dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        dtype,
        ndim,
        dims,
        NULL,
        data,
#ifdef CARMA_C_CONTIGUOUS_MODE
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#else
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#endif
        NULL
    ));

    // copy the array to a well behaved F-order
    api.PyArray_CopyInto_(dest, src);
    PyArray_ENABLEFLAGS(dest, NPY_ARRAY_OWNDATA);
    return reinterpret_cast<PyObject*>(dest);
}  // copy_array

template <typename T>
inline static PyObject* create_array(T* data, size_t ndim, size_t nelem, size_t nrows, size_t ncols, size_t nslices=0) {
    auto& api = carman::npy_api::get();

    static constexpr int npy_type = get_type<T>();
    PyArray_Descr* dtype = api.PyArray_DescrFromType_(npy_type);
    Py_INCREF(dtype);

    npy_intp* dims = new npy_intp[ndim];
    dims[0] = nrows;
    dims[1] = ncols;
    if (ndim == 3) {
        dims[2] = nslices;
    }

    // build an PyArray to do F-order copy
    // strides are computed based on dims and flag
    auto dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        dtype,
        ndim,
        dims,
        NULL,
        data,
#ifdef CARMA_C_CONTIGUOUS_MODE
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#else
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
#endif
        NULL
    ));

    // dims is copied over
    delete[] dims;
    // copy the array to a well behaved F-order
    PyArray_ENABLEFLAGS(dest, NPY_ARRAY_OWNDATA);
    return reinterpret_cast<PyObject*>(dest);
}  // create_array

}  // namespace details
}  // namespace carma

#endif  // INCLUDE_CARMA_BITS_CNUMPY_HPP_
