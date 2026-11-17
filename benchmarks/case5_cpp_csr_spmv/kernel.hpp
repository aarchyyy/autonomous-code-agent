#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

// Patched C++17 Kernel: 4-Way Unrolled CSR SpMV with Tail Cleanup & Widened FP64 Accumulator
inline std::vector<float> tiled_csr_spmv_f32(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    int num_rows,
    int num_cols
) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    if (static_cast<int>(row_ptr.size()) != num_rows + 1) {
        throw std::invalid_argument("row_ptr length must equal num_rows + 1");
    }
    if (col_indices.size() != values.size()) {
        throw std::invalid_argument("col_indices and values size mismatch");
    }
    if (static_cast<int>(x.size()) != num_cols) {
        throw std::invalid_argument("Input vector x length must equal num_cols");
    }
    if (!row_ptr.empty() && row_ptr[0] != 0) {
        throw std::invalid_argument("row_ptr[0] must be 0");
    }
    for (int j = 0; j < num_cols; ++j) {
        if (!std::isfinite(x[j])) {
            throw std::invalid_argument("Non-finite element in dense vector x");
        }
    }

    const int nnz_total = static_cast<int>(values.size());
    std::vector<float> y(static_cast<std::size_t>(num_rows), 0.0f);

    for (int r = 0; r < num_rows; ++r) {
        const int start = row_ptr[r];
        const int end = row_ptr[r + 1];
        if (start < 0 || end < start || end > nnz_total) {
            throw std::out_of_range("Invalid CSR row_ptr slice bounds");
        }
        if (start == end) {
            continue;
        }

        double acc0 = 0.0;
        double acc1 = 0.0;
        int k = start;
        for (; k + 3 < end; k += 4) {
            for (int u = 0; u < 4; ++u) {
                const int col = col_indices[k + u];
                const float v = values[k + u];
                if (!std::isfinite(v)) {
                    throw std::invalid_argument("Non-finite sparse value");
                }
                if (col < 0 || col >= num_cols) {
                    throw std::out_of_range("CSR column index out of bounds");
                }
            }
            acc0 += static_cast<double>(values[k]) * static_cast<double>(x[col_indices[k]])
                  + static_cast<double>(values[k + 1]) * static_cast<double>(x[col_indices[k + 1]]);
            acc1 += static_cast<double>(values[k + 2]) * static_cast<double>(x[col_indices[k + 2]])
                  + static_cast<double>(values[k + 3]) * static_cast<double>(x[col_indices[k + 3]]);
        }

        double tail_acc = 0.0;
        for (; k < end; ++k) {
            const int col = col_indices[k];
            const float v = values[k];
            if (!std::isfinite(v)) {
                throw std::invalid_argument("Non-finite sparse value");
            }
            if (col < 0 || col >= num_cols) {
                throw std::out_of_range("CSR column index out of bounds");
            }
            tail_acc += static_cast<double>(v) * static_cast<double>(x[col]);
        }

        y[static_cast<std::size_t>(r)] = static_cast<float>((acc0 + acc1) + tail_acc);
    }

    return y;
}
