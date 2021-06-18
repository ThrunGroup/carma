/*  carma_bits/cnalloc.hpp: Wrappers around Numpy's (de)allocator
 *  Copyright (c) 2021 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */
#ifndef INCLUDE_CARMA_BITS_CNALLOC_HPP_
#define INCLUDE_CARMA_BITS_CNALLOC_HPP_

#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <cstddef>
#ifdef CARMA_DEV_DEBUG
#include <iostream>
#endif

namespace cnalloc {

inline void* npy_malloc(std::size_t bytes) {
    if (PyArray_API == NULL) {
        _import_array();
    }
#ifdef CARMA_DEV_DEBUG
    void* ptr = PyDataMem_NEW(bytes);
    std::cout << "\n-----------\nCARMA DEBUG\n-----------\n";
    std::cout << "Using numpy allocator" << "\n";
    std::cerr << "Allocated memory @" << ptr << std::endl;
    std::cout << "-----------\n";
    return ptr;
#else
    return PyDataMem_NEW(bytes);
#endif  // ARMA_EXTRA_DEBUG
} // npy_malloc

inline void npy_free(void* ptr) {
    if (PyArray_API == NULL) {
        _import_array();
    }
#ifdef CARMA_DEV_DEBUG
    std::cout << "\n-----------\nCARMA DEBUG\n-----------\n";
    std::cerr << "Using numpy deallocator\n";
    std::cerr << "Freeing memory @" << ptr << std::endl;
    std::cout << "-----------\n";
#endif  // ARMA_EXTRA_DEBUG
    PyDataMem_FREE(ptr);
} // npy_free

} // namespace cnalloc

#define ARMA_ALIEN_MEM_ALLOC_FUNCTION cnalloc::npy_malloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION cnalloc::npy_free
#ifndef CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
  #define CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
#endif
#endif  // INCLUDE_CARMA_BITS_CNALLOC_HPP_
