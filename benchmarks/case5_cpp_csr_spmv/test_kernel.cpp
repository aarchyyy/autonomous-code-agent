#include "kernel.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void assert_near(float a, float b, float tol = 1e-6f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "AssertionError: expected " + std::to_string(b) + ", got " + std::to_string(a)
        );
    }
}

void test_aligned_tile4_and_empty_row() {
    std::vector<int> row_ptr = {0, 4, 4};
    std::vector<int> cols = {0, 1, 2, 3};
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> x = {2.0f, 1.0f, 0.5f, 3.0f};
    auto y = tiled_csr_spmv_f32(row_ptr, cols, vals, x, 2, 4);
    if (y.size() != 2) throw std::runtime_error("Expected 2 output rows");
    assert_near(y[0], 17.5f);
    assert_near(y[1], 0.0f);
}

void test_unaligned_tail_elements() {
    // Row 0 has nnz=3 (<4), Row 1 has nnz=5 (4 + 1 tail)
    std::vector<int> row_ptr = {0, 3, 8};
    std::vector<int> cols = {0, 1, 2, 0, 1, 2, 3, 4};
    std::vector<float> vals = {1.5f, 2.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f};
    std::vector<float> x = {2.0f, 3.0f, 0.5f, 4.0f, 2.0f};
    auto y = tiled_csr_spmv_f32(row_ptr, cols, vals, x, 2, 5);
    assert_near(y[0], 11.0f);
    assert_near(y[1], 19.5f);
}

void test_negative_and_oob_col_index() {
    bool caught_neg = false;
    try {
        tiled_csr_spmv_f32({0, 1}, {-1}, {1.0f}, {3.0f}, 1, 1);
    } catch (const std::out_of_range&) {
        caught_neg = true;
    }
    if (!caught_neg) throw std::runtime_error("Negative column index was not rejected with out_of_range");

    bool caught_oob = false;
    try {
        tiled_csr_spmv_f32({0, 2}, {0, 9}, {1.0f, 2.0f}, {3.0f, 4.0f}, 1, 2);
    } catch (const std::out_of_range&) {
        caught_oob = true;
    }
    if (!caught_oob) throw std::runtime_error("Out-of-bounds column index was not rejected");
}

void test_malformed_csr_and_shape() {
    bool caught_monotone = false;
    try {
        tiled_csr_spmv_f32({0, 2, 1}, {0, 0}, {1.0f, 1.0f}, {2.0f}, 2, 1);
    } catch (const std::exception&) {
        caught_monotone = true;
    }
    if (!caught_monotone) throw std::runtime_error("Non-monotonic row_ptr was not rejected");

    bool caught_dim = false;
    try {
        tiled_csr_spmv_f32({0, 1}, {0}, {1.0f}, {2.0f, 3.0f}, 1, 1);
    } catch (const std::invalid_argument&) {
        caught_dim = true;
    }
    if (!caught_dim) throw std::runtime_error("Vector dimension mismatch was not rejected");
}

void test_non_finite_rejected() {
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    bool caught_nan = false;
    try {
        tiled_csr_spmv_f32({0, 1}, {0}, {nan_v}, {1.0f}, 1, 1);
    } catch (const std::invalid_argument&) {
        caught_nan = true;
    }
    if (!caught_nan) throw std::runtime_error("NaN sparse value was not rejected");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> suite = {
        {"test_aligned_tile4_and_empty_row", test_aligned_tile4_and_empty_row},
        {"test_unaligned_tail_elements", test_unaligned_tail_elements},
        {"test_negative_and_oob_col_index", test_negative_and_oob_col_index},
        {"test_malformed_csr_and_shape", test_malformed_csr_and_shape},
        {"test_non_finite_rejected", test_non_finite_rejected},
    };

    int passed = 0;
    int failed = 0;
    for (const auto& [name, fn] : suite) {
        try {
            fn();
            std::cerr << "[PASS] " << name << "\n";
            ++passed;
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << name << ": " << ex.what() << "\n";
            ++failed;
        }
    }
    std::cerr << "[SUMMARY] tests_run=" << suite.size()
              << " passed=" << passed
              << " failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}
