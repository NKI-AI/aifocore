// Copyright 2025 Jonas Teuwen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aifocore/math/ndarray_all.h"

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace aifocore::math {
namespace {

// Test default construction
TEST(NDArrayTest, DefaultConstruction) {
  NDArray<int, 2> arr;
  EXPECT_TRUE(arr.Empty());
  EXPECT_EQ(arr.Size(), 0);
}

// Test construction with shape
TEST(NDArrayTest, ConstructionWithShape) {
  NDArray<int, 2> arr({3, 4});
  EXPECT_FALSE(arr.Empty());
  EXPECT_EQ(arr.Size(), 12);

  auto shape = arr.Shape();
  EXPECT_EQ(shape[0], 3);
  EXPECT_EQ(shape[1], 4);
}

// Test construction with shape and init value
TEST(NDArrayTest, ConstructionWithInitValue) {
  NDArray<int, 2> arr({3, 4}, 42);
  EXPECT_EQ(arr.Size(), 12);

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(arr(i, j), 42);
    }
  }
}

// Test element access via operator()
TEST(NDArrayTest, ElementAccessOperator) {
  NDArray<int, 2> arr({3, 4});

  // Write
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr(i, j) = static_cast<int>(i * 4 + j);
    }
  }

  // Read
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(arr(i, j), static_cast<int>(i * 4 + j));
    }
  }
}

// Test element access via At()
TEST(NDArrayTest, ElementAccessAt) {
  NDArray<int, 2> arr({3, 4});

  // Write
  arr.At({0, 0}) = 10;
  arr.At({1, 2}) = 20;
  arr.At({2, 3}) = 30;

  // Read
  EXPECT_EQ(arr.At({0, 0}), 10);
  EXPECT_EQ(arr.At({1, 2}), 20);
  EXPECT_EQ(arr.At({2, 3}), 30);
}

// Test shape and strides
TEST(NDArrayTest, ShapeAndStrides) {
  NDArray<int, 3> arr({2, 3, 4});

  auto shape = arr.Shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);

  // Row-major: strides should be [12, 4, 1]
  auto strides = arr.Strides();
  EXPECT_EQ(strides[0], 12);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 1);
}

// Test NDArrayView (non-owning)
TEST(NDArrayViewTest, ViewMode) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  NDArrayView<int, 2> arr_view(data.data(), {2, 3});

  // Check shape
  auto shape = arr_view.Shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);

  // Check data access
  EXPECT_EQ(arr_view(0, 0), 1);
  EXPECT_EQ(arr_view(0, 1), 2);
  EXPECT_EQ(arr_view(1, 2), 6);

  // Modify through view
  arr_view(1, 1) = 99;
  EXPECT_EQ(data[4], 99);  // Verify original data is modified
}

// Test creating view from NDArray
TEST(NDArrayTest, CreateView) {
  NDArray<int, 2> arr({2, 3}, 42);
  arr(0, 0) = 10;
  arr(1, 2) = 20;

  // Create view
  auto view = arr.View();
  EXPECT_EQ(view(0, 0), 10);
  EXPECT_EQ(view(1, 2), 20);
  EXPECT_EQ(view(0, 1), 42);

  // Modify through view
  view(1, 1) = 99;
  EXPECT_EQ(arr(1, 1), 99);
}

// Test unary negation operator
TEST(NDArrayTest, UnaryNegation) {
  NDArray<int, 2> arr({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr(i, j) = static_cast<int>(i * 3 + j + 1);
    }
  }

  // Apply negation (lazy expression)
  auto neg_expr = -arr;

  // Check negated values via flat evaluation
  EXPECT_EQ(neg_expr.EvalFlat(0), -1);
  EXPECT_EQ(neg_expr.EvalFlat(1), -2);
  EXPECT_EQ(neg_expr.EvalFlat(5), -6);
}

// Test expression evaluation (assignment from expression)
TEST(NDArrayTest, ExpressionEvaluation) {
  NDArray<int, 2> arr({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr(i, j) = static_cast<int>(i * 3 + j + 1);
    }
  }

  // Create new array from expression (triggers flat evaluation)
  NDArray<int, 2> neg_arr = -arr;

  // Check values in new array
  EXPECT_EQ(neg_arr(0, 0), -1);
  EXPECT_EQ(neg_arr(0, 1), -2);
  EXPECT_EQ(neg_arr(0, 2), -3);
  EXPECT_EQ(neg_arr(1, 0), -4);
  EXPECT_EQ(neg_arr(1, 1), -5);
  EXPECT_EQ(neg_arr(1, 2), -6);

  // Check original is unchanged
  EXPECT_EQ(arr(0, 0), 1);
  EXPECT_EQ(arr(1, 2), 6);
}

// Test bounds checking - At() is always checked
TEST(NDArrayTest, BoundsChecking) {
  NDArray<int, 2> arr({3, 4});

  // Valid access
  EXPECT_NO_THROW(arr.At({2, 3}));

  // Out of bounds access - At() always throws
  EXPECT_THROW(arr.At({3, 0}), std::out_of_range);
  EXPECT_THROW(arr.At({0, 4}), std::out_of_range);
  EXPECT_THROW(arr.At({3, 3}), std::out_of_range);
}

// Test bounds checking on view - At() is always checked
TEST(NDArrayViewTest, BoundsChecking) {
  std::vector<int> data(12, 0);
  NDArrayView<int, 2> view(data.data(), {3, 4});

  // Valid access
  EXPECT_NO_THROW(view.At({2, 3}));

  // Out of bounds access - At() always throws
  EXPECT_THROW(view.At({3, 0}), std::out_of_range);
  EXPECT_THROW(view.At({0, 4}), std::out_of_range);
}

// Test copy constructor
TEST(NDArrayTest, CopyConstructor) {
  NDArray<int, 2> arr1({2, 3}, 42);
  arr1(0, 0) = 10;
  arr1(1, 2) = 20;

  NDArray<int, 2> arr2(arr1);

  // Check copied values
  EXPECT_EQ(arr2(0, 0), 10);
  EXPECT_EQ(arr2(1, 2), 20);
  EXPECT_EQ(arr2(0, 1), 42);

  // Modify copy and check original is unchanged
  arr2(0, 0) = 99;
  EXPECT_EQ(arr1(0, 0), 10);
  EXPECT_EQ(arr2(0, 0), 99);
}

// Test move constructor
TEST(NDArrayTest, MoveConstructor) {
  NDArray<int, 2> arr1({2, 3}, 42);
  arr1(0, 0) = 10;

  NDArray<int, 2> arr2(std::move(arr1));

  // Check moved values
  EXPECT_EQ(arr2(0, 0), 10);
  EXPECT_EQ(arr2(0, 1), 42);

  // Original should be empty
  EXPECT_TRUE(arr1.Empty());
}

// Test copy assignment
TEST(NDArrayTest, CopyAssignment) {
  NDArray<int, 2> arr1({2, 3}, 42);
  arr1(0, 0) = 10;

  NDArray<int, 2> arr2;
  arr2 = arr1;

  // Check assigned values
  EXPECT_EQ(arr2(0, 0), 10);
  EXPECT_EQ(arr2(0, 1), 42);

  // Modify and check independence
  arr2(0, 0) = 99;
  EXPECT_EQ(arr1(0, 0), 10);
}

// Test move assignment
TEST(NDArrayTest, MoveAssignment) {
  NDArray<int, 2> arr1({2, 3}, 42);
  arr1(0, 0) = 10;

  NDArray<int, 2> arr2;
  arr2 = std::move(arr1);

  // Check moved values
  EXPECT_EQ(arr2(0, 0), 10);
  EXPECT_EQ(arr2(0, 1), 42);

  // Original should be empty
  EXPECT_TRUE(arr1.Empty());
}

// Test assignment from expression
TEST(NDArrayTest, AssignmentFromExpression) {
  NDArray<int, 2> arr1({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr1(i, j) = static_cast<int>(i * 3 + j + 1);
    }
  }

  NDArray<int, 2> arr2;
  arr2 = -arr1;

  // Check assigned negated values
  EXPECT_EQ(arr2(0, 0), -1);
  EXPECT_EQ(arr2(1, 2), -6);
}

// Test 3D array
TEST(NDArrayTest, ThreeDimensional) {
  NDArray<int, 3> arr({2, 3, 4});

  // Check size and shape
  EXPECT_EQ(arr.Size(), 24);
  auto shape = arr.Shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);

  // Write and read
  arr(0, 0, 0) = 1;
  arr(1, 2, 3) = 999;
  EXPECT_EQ(arr(0, 0, 0), 1);
  EXPECT_EQ(arr(1, 2, 3), 999);
}

// Test 4D array
TEST(NDArrayTest, FourDimensional) {
  NDArray<double, 4> arr({2, 3, 4, 5}, 3.14);

  // Check size
  EXPECT_EQ(arr.Size(), 120);

  // Check init value
  EXPECT_DOUBLE_EQ(arr(0, 0, 0, 0), 3.14);
  EXPECT_DOUBLE_EQ(arr(1, 2, 3, 4), 3.14);

  // Modify and check
  arr(1, 1, 1, 1) = 2.71;
  EXPECT_DOUBLE_EQ(arr(1, 1, 1, 1), 2.71);
}

// Test flat data access
TEST(NDArrayTest, FlatDataAccess) {
  NDArray<int, 2> arr({2, 3});

  // Fill via flat access
  int* data = arr.Data();
  for (std::size_t i = 0; i < arr.Size(); ++i) {
    data[i] = static_cast<int>(i + 1);
  }

  // Verify via 2D access
  EXPECT_EQ(arr(0, 0), 1);
  EXPECT_EQ(arr(0, 1), 2);
  EXPECT_EQ(arr(0, 2), 3);
  EXPECT_EQ(arr(1, 0), 4);
  EXPECT_EQ(arr(1, 1), 5);
  EXPECT_EQ(arr(1, 2), 6);
}

// Test with float type
TEST(NDArrayTest, FloatType) {
  NDArray<float, 2> arr({2, 2}, 1.5f);

  arr(0, 0) = 2.5f;
  arr(1, 1) = 3.5f;

  EXPECT_FLOAT_EQ(arr(0, 0), 2.5f);
  EXPECT_FLOAT_EQ(arr(0, 1), 1.5f);
  EXPECT_FLOAT_EQ(arr(1, 1), 3.5f);

  // Test negation with float
  NDArray<float, 2> neg = -arr;
  EXPECT_FLOAT_EQ(neg(0, 0), -2.5f);
  EXPECT_FLOAT_EQ(neg(1, 1), -3.5f);
}

// Test flat evaluation performance
TEST(NDArrayTest, FlatEvaluationPath) {
  // Large array to verify flat evaluation is used
  NDArray<int, 2> arr({100, 100}, 1);

  // Negate via expression
  NDArray<int, 2> neg = -arr;

  // Verify correctness
  EXPECT_EQ(neg(0, 0), -1);
  EXPECT_EQ(neg(99, 99), -1);
  EXPECT_EQ(neg.Size(), 10000);
}

// Test UnsafeAt accessor
TEST(NDArrayTest, UnsafeAccessor) {
  NDArray<int, 2> arr({3, 4}, 42);
  arr(1, 2) = 99;

  // UnsafeAt should work without bounds checking
  EXPECT_EQ(arr.UnsafeAt({1, 2}), 99);
  EXPECT_EQ(arr.UnsafeAt({0, 0}), 42);
}

// Test DataUnsafe accessor
TEST(NDArrayTest, UnsafeDataAccess) {
  NDArray<int, 2> arr({2, 3});
  int* data = arr.DataUnsafe();
  for (std::size_t i = 0; i < arr.Size(); ++i) {
    data[i] = static_cast<int>(i + 10);
  }

  EXPECT_EQ(arr(0, 0), 10);
  EXPECT_EQ(arr(0, 1), 11);
  EXPECT_EQ(arr(1, 2), 15);
}

// Test Rank() static method
TEST(NDArrayTest, RankMethod) {
  NDArray<int, 2> arr2d({3, 4});
  NDArray<int, 3> arr3d({2, 3, 4});

  EXPECT_EQ(arr2d.Rank(), 2);
  EXPECT_EQ(arr3d.Rank(), 3);

  // Also test on expressions
  auto expr = -arr2d;
  EXPECT_EQ(expr.Rank(), 2);
}

TEST(NDArrayViewTest, ViewInExpressions) {
  NDArray<int, 2> base({2, 2});
  base(0, 0) = 1;
  base(0, 1) = 2;
  base(1, 0) = 3;
  base(1, 1) = 4;

  auto v = base.View();

  NDArray<int, 2> sum = v + v;
  EXPECT_EQ(sum(0, 0), 2);
  EXPECT_EQ(sum(1, 1), 8);
}

// Note: Rvalue View() and operator-() are deleted at compile time.
// Attempting NDArray<int,2>({2,2}).View() or std::move(arr).View()
// will fail to compile, which is tested via compilation failures elsewhere.

TEST(NDArrayViewTest, ConstViewAndExpression) {
  NDArray<int, 2> base({2, 2});
  base(0, 0) = 1;
  base(0, 1) = 2;
  base(1, 0) = 3;
  base(1, 1) = 4;

  const auto view = base.View();
  auto neg = -view;
  NDArray<int, 2> res = neg;

  EXPECT_EQ(res(0, 0), -1);
  EXPECT_EQ(res(1, 1), -4);
}

TEST(NDArrayTest, ZeroSizeShape) {
  NDArray<int, 1> arr({0});
  EXPECT_TRUE(arr.Empty());
  EXPECT_EQ(arr.Size(), 0);
}

// Test zero-copy and lazy evaluation with a tracked type
TEST(NDArrayTest, ZeroCopyAndLazyEvaluation) {
  // Counter to track evaluation
  static int eval_count = 0;

  struct TrackedInt {
    int value;

    TrackedInt() : value(0) {}

    explicit TrackedInt(int v) : value(v) {}

    // Track when operations actually execute
    TrackedInt operator-() const {
      eval_count++;
      return TrackedInt(-value);
    }

    TrackedInt operator+(const TrackedInt& other) const {
      eval_count++;
      return TrackedInt(value + other.value);
    }

    bool operator>(const TrackedInt& other) const {
      return value > other.value;
    }

    bool operator<(const TrackedInt& other) const {
      return value < other.value;
    }
  };

  eval_count = 0;

  NDArray<TrackedInt, 2> a({2, 2}, TrackedInt(5));
  NDArray<TrackedInt, 2> b({2, 2}, TrackedInt(3));

  // Create expression - should NOT evaluate yet (lazy)
  auto expr = a + b;
  EXPECT_EQ(eval_count, 0)
      << "Expression creation should be lazy (no evaluation)";

  // Chain more operations - still should NOT evaluate
  auto expr2 = expr + a;
  EXPECT_EQ(eval_count, 0) << "Expression chaining should be lazy";

  // Add negated array - still lazy
  auto expr3 = expr2 + (-a);
  EXPECT_EQ(eval_count, 0) << "More chaining should remain lazy";

  // Only when we assign to an NDArray should evaluation happen
  eval_count = 0;  // Reset counter
  NDArray<TrackedInt, 2> result = expr3;

  // Should have evaluated: (a + b) + a + (-a) for 4 elements
  // Each element: 3 additions + 1 negation = 4 ops * 4 elements = 16 ops
  EXPECT_EQ(eval_count, 16)
      << "Evaluation should happen exactly once on assignment";

  // Verify correctness: (5 + 3) + 5 + (-5) = 8
  EXPECT_EQ(result(0, 0).value, 8);
}

// Test that NDArray is stored by const& (not copied) in expressions
TEST(NDArrayTest, NDArrayStoredByReference) {
  // Track copy construction
  static int copy_count = 0;
  static int move_count = 0;

  struct CopyTracker {
    int value;

    CopyTracker() : value(0) {}

    explicit CopyTracker(int v) : value(v) {}

    CopyTracker(const CopyTracker& other) : value(other.value) { copy_count++; }

    CopyTracker(CopyTracker&& other) noexcept : value(other.value) {
      move_count++;
      other.value = 0;
    }

    CopyTracker& operator=(const CopyTracker& other) {
      value = other.value;
      copy_count++;
      return *this;
    }

    CopyTracker& operator=(CopyTracker&& other) noexcept {
      value = other.value;
      move_count++;
      other.value = 0;
      return *this;
    }

    CopyTracker operator+(const CopyTracker& other) const {
      return CopyTracker(value + other.value);
    }

    CopyTracker operator-() const { return CopyTracker(-value); }

    bool operator>(const CopyTracker& other) const {
      return value > other.value;
    }

    bool operator<(const CopyTracker& other) const {
      return value < other.value;
    }
  };

  copy_count = 0;
  move_count = 0;

  // Create arrays (will do some copies during construction)
  NDArray<CopyTracker, 2> a({2, 2}, CopyTracker(5));
  NDArray<CopyTracker, 2> b({2, 2}, CopyTracker(3));

  int copies_after_construction = copy_count;

  // Create expression - should NOT copy the NDArray data
  auto expr = a + b;
  EXPECT_EQ(copy_count, copies_after_construction)
      << "Creating expression should not copy NDArray (stored by const&)";

  // Chain operations - should still not copy NDArray data
  auto expr2 = Max(expr, a);
  EXPECT_EQ(copy_count, copies_after_construction)
      << "Chaining should not copy NDArray";

  // Only evaluation into result should copy
  copy_count = 0;  // Reset to count only the evaluation
  NDArray<CopyTracker, 2> result = expr2;

  // Result construction will copy elements during evaluation
  // 4 elements should be copied/moved into result
  EXPECT_GT(copy_count + move_count, 0)
      << "Final evaluation should copy/move elements into result";
}

// Test expression size is small (proves lightweight storage)
TEST(NDArrayTest, ExpressionTypesAreLightweight) {
  NDArray<int, 2> a({10, 10}, 1);
  NDArray<int, 2> b({10, 10}, 2);

  // Expression types should be small (just pointers + metadata)
  auto unary_expr = -a;
  auto binary_expr = a + b;
  auto scalar_expr = Max(a, 5);
  auto composed = Max(a + b, -a);

  // Use the composed expression to avoid unused variable warning
  (void)composed;

  // NDArray itself contains vector (shape, strides, std::vector<int>)
  // std::vector is ~24 bytes on most platforms, plus shape/strides
  size_t ndarray_size = sizeof(a);

  // Expressions should be significantly smaller than NDArray
  // (they hold pointers/references, not data)
  EXPECT_LT(sizeof(unary_expr), ndarray_size)
      << "Unary expr should be smaller than NDArray";
  EXPECT_LT(sizeof(binary_expr), ndarray_size * 2)
      << "Binary expr should be smaller than 2x NDArray";
  EXPECT_LT(sizeof(scalar_expr), ndarray_size)
      << "Scalar expr should be smaller than NDArray";

  // More importantly: binary_expr should NOT contain full copies of NDArray
  // If it stored by value, sizeof would be ~2*sizeof(NDArray) + overhead
  // With const&, sizeof is just ~2 pointers + metadata
  // Expect: 2 refs (16 bytes) + shape (16) + size (8) + op (1) ~= 50 bytes
  EXPECT_LT(sizeof(binary_expr), 150)
      << "Binary expr proves const& storage (not copying NDArray data)";
}

}  // namespace
}  // namespace aifocore::math
