import unittest
from kernel import gqa_kv_broadcast_kernel

class TestGQAKVBroadcast(unittest.TestCase):
    def test_gqa_expansion(self):
        # 1 batch, 2 KV heads, seq_len=1, head_dim=1
        kv_input = [[[10.0]], [[20.0]]]  # shape [1, 2, 1, 1]
        res = gqa_kv_broadcast_kernel([kv_input], num_query_heads=12, num_kv_heads=2)
        
        # Expected: 12 heads per batch item
        self.assertEqual(len(res[0]), 12)
        # First 6 heads should match KV head 0: [[10.0]]
        for h in range(6):
            self.assertEqual(res[0][h], [[10.0]])
        # Remaining 6 heads should match KV head 1: [[20.0]]
        for h in range(6, 12):
            self.assertEqual(res[0][h], [[20.0]])

if __name__ == "__main__":
    unittest.main()
