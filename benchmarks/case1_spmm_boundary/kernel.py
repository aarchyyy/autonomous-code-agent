# Patched Kernel: SpMM CSR with Strict Bounds & Compensated Accumulation
import math


def spmm_csr_operator(row_ptr, col_indices, values, dense_b, num_rows, num_cols):
    """
    Sparse Matrix-Matrix Multiplication (C = A_csr * B_dense).
    Enforces CSR structure invariants, negative/out-of-range index guards,
    empty-row fast paths, and math.fsum accumulation.
    """
    if num_rows < 0 or num_cols < 0:
        raise ValueError("Matrix dimensions must be non-negative")
    if len(row_ptr) != num_rows + 1:
        raise ValueError(f"row_ptr length {len(row_ptr)} != num_rows + 1 ({num_rows + 1})")
    if len(col_indices) != len(values):
        raise ValueError("col_indices and values length mismatch")
    if row_ptr and row_ptr[0] != 0:
        raise ValueError("row_ptr must start at 0")

    num_b_rows = len(dense_b)
    for row_b in dense_b:
        if len(row_b) != num_cols:
            raise ValueError("dense_b column dimension mismatch")

    result = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]

    for r in range(num_rows):
        start_idx = row_ptr[r]
        end_idx = row_ptr[r + 1]
        if start_idx < 0 or end_idx < start_idx or end_idx > len(col_indices):
            raise IndexError(f"Invalid CSR row_ptr slice [{start_idx}, {end_idx}) at row {r}")
        if start_idx == end_idx:
            continue

        col_terms = [[] for _ in range(num_cols)]
        for k_idx in range(start_idx, end_idx):
            c_idx = col_indices[k_idx]
            val = values[k_idx]
            if not math.isfinite(val):
                raise ValueError("Non-finite sparse matrix value")
            if not (0 <= c_idx < num_b_rows):
                raise IndexError(f"Column index {c_idx} out of bounds for dense_b with {num_b_rows} rows")
            row_b = dense_b[c_idx]
            for j in range(num_cols):
                b_val = row_b[j]
                if not math.isfinite(b_val):
                    raise ValueError("Non-finite dense matrix value")
                col_terms[j].append(val * b_val)

        for j in range(num_cols):
            result[r][j] = math.fsum(col_terms[j])

    return result
