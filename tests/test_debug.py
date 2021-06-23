import sys
import numpy as np
import test_carma as tc
import test_carma as carma

sample = np.random.normal(0, 1, (1000, 2))
fsample = np.asarray(sample, order='F')
ref_sample = sample.sum()

# print('------ C-order -- Borrow ------')
# print('ref count sample -- pre: ', sys.getrefcount(sample))
# print('ref count fsample -- pre: ', sys.getrefcount(fsample))
# arr = tc.debug_arr_to_mat(sample, 0)
# print('ref count sample -- post: ', sys.getrefcount(sample))
# print('ref count fsample -- post: ', sys.getrefcount(fsample))
# assert np.isclose(ref_sample, arr.sum())
#
# print('------ C-order -- Copy ------')
# print('ref count sample -- pre: ', sys.getrefcount(sample))
# print('ref count fsample -- pre: ', sys.getrefcount(fsample))
# arr = tc.debug_arr_to_mat(sample, 1)
# print('ref count sample -- post: ', sys.getrefcount(sample))
# print('ref count fsample -- post: ', sys.getrefcount(fsample))
# assert np.isclose(ref_sample, arr.sum())
#
# print('------ C-order -- Steal ------')
# print('ref count sample -- pre: ', sys.getrefcount(sample))
# print('ref count fsample -- pre: ', sys.getrefcount(fsample))
# arr = tc.debug_arr_to_mat(sample, -1)
# print('ref count sample -- post: ', sys.getrefcount(sample))
# print('ref count fsample -- post: ', sys.getrefcount(fsample))
# assert np.isclose(ref_sample, arr.sum())
#
# print('------ F-order -- Borrow ------')
# print('ref count sample -- pre: ', sys.getrefcount(sample))
# print('ref count fsample -- pre: ', sys.getrefcount(fsample))
# arr = tc.debug_arr_to_mat(fsample, 0)
# print('ref count sample -- post: ', sys.getrefcount(sample))
# print('ref count fsample -- post: ', sys.getrefcount(fsample))
# assert np.isclose(ref_sample, arr.sum())
#
# print('------ F-order -- Copy ------')
# print('ref count sample -- pre: ', sys.getrefcount(sample))
# print('ref count fsample -- pre: ', sys.getrefcount(fsample))
# arr = tc.debug_arr_to_mat(fsample, 1)
# print('ref count sample -- post: ', sys.getrefcount(sample))
# print('ref count fsample -- post: ', sys.getrefcount(fsample))
# assert np.isclose(ref_sample, arr.sum())
#
# print('------ F-order -- Steal ------')
# print('ref count sample -- pre: ', sys.getrefcount(sample))
# print('ref count fsample -- pre: ', sys.getrefcount(fsample))
# arr = tc.debug_arr_to_mat(fsample, -1)
# print('ref count sample -- post: ', sys.getrefcount(sample))
# print('ref count fsample -- post: ', sys.getrefcount(fsample))
# assert np.isclose(ref_sample, arr.sum())
def test_mat_to_arr_plus_one():
    print('-------------------------------------')
    print('mat_to_arr_plus_one')
    print('-------------------------------------')
    sample = np.asarray(
        np.random.normal(size=(20, 2)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.mat_to_arr_plus_one(sample, False)
    assert np.allclose(mat, sample + 1)

def test_mat_roundtrip_default():
    print('-------------------------------------')
    print('Default')
    print('-------------------------------------')
    og_sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy(order='F')
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)

def test_mat_roundtrip_large():
    print('-------------------------------------')
    print('Large')
    print('-------------------------------------')
    og_sample = np.asarray(
        np.random.normal(size=(1000, 1000)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy(order='F')
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)

def test_mat_roundtrip_small():
    print('-------------------------------------')
    print('Small')
    print('-------------------------------------')
    og_sample = np.asarray(
        np.random.normal(size=(3, 3)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy(order='F')
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)
