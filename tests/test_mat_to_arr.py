"""Test armadillo matrix to  numpy array functions."""
import numpy as np

import test_carma as carma

###############################################################################
#                                 MAT
###############################################################################


def test_mat_to_arr():
    """Test mat_to_arr."""
    arr = carma.mat_to_arr(False)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_mat_to_arr_copy():
    """Test mat_to_arr."""
    arr = carma.mat_to_arr(True)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_mat_to_arr_plus_one():
    """Test mat_to_arr with c++ addition."""
    sample = np.asarray(
        np.random.normal(size=(20, 2)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.mat_to_arr_plus_one(sample, False)
    assert np.allclose(mat, sample + 1)


def test_mat_to_arr_plus_one_copy():
    """Test mat_to_arr with c++ addition."""
    sample = np.asarray(
        np.random.normal(size=(20, 2)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.mat_to_arr_plus_one(sample, True)
    assert np.allclose(mat, sample + 1)

###############################################################################
#                                 ROW
###############################################################################


def test_row_to_arr():
    """Test row_to_arr."""
    arr = carma.row_to_arr(False)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_row_to_arr_copy():
    """Test row_to_arr."""
    arr = carma.row_to_arr(True)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_row_to_arr_plus_one():
    """Test row_to_arr with c++ addition."""
    sample = np.asarray(
        np.random.normal(size=(20)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.row_to_arr_plus_one(sample, False)
    assert np.allclose(mat, sample + 1)


def test_row_to_arr_plus_one_copy():
    """Test row_to_arr with c++ addition."""
    sample = np.asarray(
        np.random.normal(size=(20)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.row_to_arr_plus_one(sample, True)
    assert np.allclose(mat, sample + 1)


###############################################################################
#                                 COL
###############################################################################


def test_col_to_arr():
    """Test col_to_arr."""
    arr = carma.col_to_arr(False)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_col_to_arr_copy():
    """Test col_to_arr."""
    arr = carma.col_to_arr(True)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_col_to_arr_plus_one():
    """Test col_to_arr with c++ addition."""
    sample = np.asarray(
        np.random.normal(size=(20, 1)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.col_to_arr_plus_one(sample, False)
    assert np.allclose(mat, sample + 1)


def test_col_to_arr_plus_one_copy():
    """Test col_to_arr with c++ addition."""
    sample = np.asarray(
        np.random.normal(size=(20, 1)),
        dtype=np.float64,
        order='F'
    )
    mat = carma.col_to_arr_plus_one(sample, True)
    assert np.allclose(mat, sample + 1)

###############################################################################
#                                 CUBE
###############################################################################


def test_cube_to_arr():
    """Test cube_to_arr."""
    arr = carma.cube_to_arr(False)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-16).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_cube_to_arr_return():
    """Test cube_to_arr."""
    arr = carma.cube_to_arr(True)

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'

    near_null = (np.abs(arr) <= 1e-6).sum()
    assert near_null == 0, 'Conversion introduced near zero values'


def test_cube_to_arr_plus_one():
    """Test cube_to_arr with c++ addition."""
    or_sample = np.random.normal(size=(20, 3, 2))
    sample = np.asarray(
        or_sample,
        dtype=np.float64,
        order='F'
    )
    mat = carma.cube_to_arr_plus_one(sample, False)
    assert np.allclose(mat, 1 + sample)


def test_cube_to_arr_plus_one_copy():
    """Test cube_to_arr with c++ addition."""
    or_sample = np.random.normal(size=(20, 3, 2))
    sample = np.asarray(
        or_sample,
        dtype=np.float64,
        order='F'
    )
    mat = carma.cube_to_arr_plus_one(sample, True)
    assert np.allclose(mat, 1 + sample)
