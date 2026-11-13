import struct
import unittest
from kernel import vector_scale_kernel


def _f32(x: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(x)))[0]


def _f32_bits(x: float) -> int:
    val = struct.unpack("!I", struct.pack("!f", float(x)))[0]
    return val if val < 0x80000000 else 0x80000000 - val


def ulp_distance_f32(a: float, b: float) -> int:
    if a == 0.0 and b == 0.0:
        return 0
    return abs(_f32_bits(a) - _f32_bits(b))


class TestVectorScale(unittest.TestCase):
    def test_basic_addition(self):
        res = vector_scale_kernel([1.0, 2.0], [3.0, 4.0], scale=1.0)
        self.assertEqual(res, [4.0, 6.0])

    def test_scaled_addition(self):
        res = vector_scale_kernel([1.0, 2.0], [3.0, 4.0], scale=2.5)
        self.assertEqual(res, [10.0, 15.0])

    def test_sub_micro_precision_not_truncated(self):
        # Decimal round(..., 6) collapses 3e-7 to 0.0 and fails this check
        a = [1e-7, -2e-7]
        b = [1e-7, -1e-7]
        scale = 1.5
        res = vector_scale_kernel(a, b, scale=scale)
        expected = [_f32(_f32(_f32(x) + _f32(y)) * _f32(scale)) for x, y in zip(a, b)]
        for got, exp in zip(res, expected):
            self.assertEqual(ulp_distance_f32(got, exp), 0)

    def test_ieee754_float32_zero_ulp(self):
        a = [0.1, 0.2, 1.337, -123.456]
        b = [0.2, 0.3, -0.421, 0.000125]
        scale = 0.7
        res = vector_scale_kernel(a, b, scale=scale)
        expected = [_f32(_f32(_f32(x) + _f32(y)) * _f32(scale)) for x, y in zip(a, b)]
        for got, exp in zip(res, expected):
            self.assertEqual(ulp_distance_f32(got, exp), 0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            vector_scale_kernel([1.0], [2.0, 3.0])

    def test_non_finite_guard(self):
        with self.assertRaises(ValueError):
            vector_scale_kernel([float("nan")], [1.0], scale=1.0)
        with self.assertRaises(ValueError):
            vector_scale_kernel([1.0], [2.0], scale=float("inf"))

    def test_empty_vectors(self):
        self.assertEqual(vector_scale_kernel([], [], scale=3.0), [])


if __name__ == "__main__":
    unittest.main()
