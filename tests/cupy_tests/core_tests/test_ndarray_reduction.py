import unittest

import cupy
from cupy import testing


@testing.gpu
class TestArrayReduction(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        return a.max()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        return a.max()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.min()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        return a.min()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        return a.min()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype)
        return a.min()


# This class compares CUB results against NumPy's
@unittest.skipIf(cupy.cuda.cub_enabled is False, 'The CUB module is not built')
@testing.parameterize(*testing.product({
    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
}))
@testing.gpu
class TestCUBreduction(unittest.TestCase):
    @testing.for_contiguous_axes()
    @testing.for_dtypes('bhilBHILefdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_min(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.min(axis=axis)

    @testing.for_contiguous_axes()
    @testing.for_dtypes('bhilBHILefdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_max(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.max(axis=axis)
