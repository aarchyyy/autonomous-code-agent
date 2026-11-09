import unittest
from kernel import spmm_csr_operator

class TestSpMMBoundary(unittest.TestCase):
    def test_standard_csr(self):
        # 2x2 sparse matrix multiplied by 2x2 dense matrix
        row_ptr = [0, 1, 2]
        col_indices = [0, 1]
        values = [2.0, 3.0]
        dense_b = [[1.0, 2.0], [3.0, 4.0]]
        res = spmm_csr_operator(row_ptr, col_indices, values, dense_b, 2, 2)
        expected = [[2.0, 4.0], [9.0, 12.0]]
        self.assertEqual(res, expected)

    def test_empty_row_skipping(self):
        # Row 1 is empty (NNZ=0): row_ptr[1] == row_ptr[2]
        row_ptr = [0, 1, 1, 2]
        col_indices = [0, 1]
        values = [5.0, 4.0]
        dense_b = [[2.0, 1.0], [1.0, 3.0]]
        res = spmm_csr_operator(row_ptr, col_indices, values, dense_b, 3, 2)
        expected = [[10.0, 5.0], [0.0, 0.0], [4.0, 12.0]]
        self.assertEqual(res, expected)

    def test_col_index_boundary(self):
        # Test out-of-range col index guard
        row_ptr = [0, 2]
        col_indices = [0, 999]  # 999 is invalid, should be safely ignored
        values = [1.0, 10.0]
        dense_b = [[5.0, 6.0]]
        res = spmm_csr_operator(row_ptr, col_indices, values, dense_b, 1, 2)
        expected = [[5.0, 6.0]]
        self.assertEqual(res, expected)

if __name__ == "__main__":
    unittest.main()
