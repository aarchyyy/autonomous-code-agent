#include "kernel.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void test_ill_conditioned_cancellation() {
    // Naive float32 left-to-right sum computes (1e8f + 1.0f) - 1e8f + 2.5f == 2.5f (loses 1.0f)
    std::vector<int> row_ptr = {0, 4};
    std::vector<int> cols = {0, 1, 2, 3};
    std::vector<float> vals = {1.0e8f, 1.0f, -1.0e8f, 2.5f};
    std::vector<float> x = {1.0f, 1.0f, 1.0f, 1.0f};
    auto y = tiled_csr_spmv_f32(row_ptr, cols, vals, x, 1, 4);
    if (y[0] != 3.5f) {
        throw std::runtime_error(
            "Holdout precision loss: expected exact 3.5f, got " + std::to_string(y[0])
        );
    }
}

void test_longtail_skewed_row_259() {
    // 259 non-zeros (64 tiles of 4 + 3 tail elements)
    const int nnz = 259;
    std::vector<int> row_ptr = {0, nnz, nnz};
    std::vector<int> cols(nnz);
    std::vector<float> vals(nnz, 0.25f);
    std::vector<float> x(nnz, 2.0f);
    for (int i = 0; i < nnz; ++i) cols[i] = i;

    auto y = tiled_csr_spmv_f32(row_ptr, cols, vals, x, 2, nnz);
    if (std::fabs(y[0] - 129.5f) > 1e-5f || y[1] != 0.0f) {
        throw std::runtime_error("Long-tail skewed row result mismatch");
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> suite = {
        {"test_ill_conditioned_cancellation", test_ill_conditioned_cancellation},
        {"test_longtail_skewed_row_259", test_longtail_skewed_row_259},
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
