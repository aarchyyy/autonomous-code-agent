import unittest
from kernel import gqa_kv_broadcast_kernel


class TestGQAKVBroadcast(unittest.TestCase):
    def test_gqa_expansion(self):
        kv_input = [[[10.0]], [[20.0]]]
        res = gqa_kv_broadcast_kernel([kv_input], num_query_heads=12, num_kv_heads=2)
        self.assertEqual(len(res[0]), 12)
        for h in range(6):
            self.assertEqual(res[0][h], [[10.0]])
        for h in range(6, 12):
            self.assertEqual(res[0][h], [[20.0]])

    def test_multi_batch_multi_seq(self):
        kv_batch = [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            [
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ],
        ]
        res = gqa_kv_broadcast_kernel(kv_batch, num_query_heads=8, num_kv_heads=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 8)
        self.assertEqual(res[0][3], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(res[1][4], [[13.0, 14.0], [15.0, 16.0]])

    def test_head_buffer_isolation(self):
        kv_input = [[[10.0, 11.0]], [[20.0, 21.0]]]
        res = gqa_kv_broadcast_kernel([kv_input], num_query_heads=4, num_kv_heads=2)
        res[0][0][0][0] = -999.0
        self.assertEqual(res[0][1][0][0], 10.0)
        self.assertEqual(kv_input[0][0][0], 10.0)

    def test_invalid_head_counts(self):
        with self.assertRaises(ValueError):
            gqa_kv_broadcast_kernel([[[[1.0]]]], num_query_heads=10, num_kv_heads=3)
        with self.assertRaises(ValueError):
            gqa_kv_broadcast_kernel([[[[1.0]]]], num_query_heads=4, num_kv_heads=0)
        with self.assertRaises(ValueError):
            gqa_kv_broadcast_kernel([[[[1.0]]]], num_query_heads=4, num_kv_heads=2)


if __name__ == "__main__":
    unittest.main()
