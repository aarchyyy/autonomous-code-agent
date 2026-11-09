import unittest
from kernel import vector_scale_kernel

class TestVectorScale(unittest.TestCase):
    def test_basic_addition(self):
        res = vector_scale_kernel([1.0, 2.0], [3.0, 4.0], scale=1.0)
        self.assertEqual(res, [4.0, 6.0])

    def test_scaled_addition(self):
        # Strict 0-ULP scaling check
        res = vector_scale_kernel([1.0, 2.0], [3.0, 4.0], scale=2.5)
        self.assertEqual(res, [10.0, 15.0])

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            vector_scale_kernel([1.0], [2.0, 3.0])

if __name__ == "__main__":
    unittest.main()
