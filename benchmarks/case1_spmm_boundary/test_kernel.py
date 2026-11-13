import unittest
from kernel import spmm_csr_operator


class TestSpMMBoundary(unittest.TestCase):
    def test_standard_csr(self):
        row_ptr = [0, 1, 2]
        col_indices = [0, 1]
        values = [2.0, 3.0]
        dense_b = [[1.0, 2.0], [3.0, 4.0]]
        res = spmm_csr_operator(row_ptr, col_indices, values, dense_b, 2, 2)
        self.assertEqual(res, [[2.0, 4.0], [9.0, 12.0]])

    def test_empty_row_skipping(self):
        row_ptr = [0, 1, 1, 2]
        col_indices = [0, 1]
        values = [5.0, 4.0]
        dense_b = [[2.0, 1.0], [1.0, 3.0]]
        res = spmm_csr_operator(row_ptr, col_indices, values, dense_b, 3, 2)
        self.assertEqual(res, [[10.0, 5.0], [0.0, 0.0], [4.0, 12.0]])

    def test_negative_col_index_rejected(self):
        # Negative index must NOT wrap around via Python dense_b[-1]
        row_ptr = [0, 1]
        col_indices = [-1]
        values = [2.0]
        dense_b = [[5.0, 6.0]]
        with self.assertRaises((IndexError, ValueError)):
            spmm_csr_operator(row_ptr, col_indices, values, dense_b, 1, 2)

    def test_out_of_range_col_index_rejected(self):
        row_ptr = [0, 2]
        col_indices = [0, 999]
        values = [1.0, 10.0]
        dense_b = [[5.0, 6.0]]
        with self.assertRaises((IndexError, ValueError)):
            spmm_csr_operator(row_ptr, col_indices, values, dense_b, 1, 2)

    def test_malformed_row_ptr_rejected(self):
        with self.assertRaises((ValueError, IndexError)):
            spmm_csr_operator([0, 2, 1], [0, 1], [1.0, 2.0], [[1.0], [2.0]], 2, 1)
        with self.assertRaises((ValueError, IndexError)):
            spmm_csr_operator([0, 1], [0], [1.0], [[1.0]], 2, 1)

    def test_compensated_float_accumulation(self):
        # Naive left-to-right float64 summation loses 1.0 against 1e16: (1e16 + 1.0) - 1e16 == 0.0
        row_ptr = [0, 3]
        col_indices = [0, 1, 2]
        values = [1e16, 1.0, -1e16]
        dense_b = [[1.0], [1.0], [1.0]]
        res = spmm_csr_operator(row_ptr, col_indices, values, dense_b, 1, 1)
        self.assertEqual(res[0][0], 1.0)


if __name__ == "__main__":
    unittest.main()
