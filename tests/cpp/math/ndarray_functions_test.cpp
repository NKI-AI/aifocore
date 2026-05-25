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

#include <stdexcept>

#include "gtest/gtest.h"

namespace aifocore::math {
namespace {

// Test binary Max operation
TEST(NDArrayFunctionsTest, BinaryMax) {
  NDArray<int, 2> arr1({2, 3});
  NDArray<int, 2> arr2({2, 3});

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr1(i, j) = static_cast<int>(i * 3 + j);
      arr2(i, j) = static_cast<int>(5 - (i * 3 + j));
    }
  }

  // Max of two arrays
  NDArray<int, 2> result = Max(arr1, arr2);

  EXPECT_EQ(result(0, 0), 5);  // max(0, 5)
  EXPECT_EQ(result(0, 1), 4);  // max(1, 4)
  EXPECT_EQ(result(1, 0), 3);  // max(3, 2)
  EXPECT_EQ(result(1, 2), 5);  // max(5, 0)
}

// Test Max with scalar
TEST(NDArrayFunctionsTest, MaxWithScalar) {
  NDArray<int, 2> arr({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr(i, j) = static_cast<int>(i * 3 + j - 2);  // -2, -1, 0, 1, 2, 3
    }
  }

  // Clamp to minimum of 0
  NDArray<int, 2> clamped = Max(arr, 0);

  EXPECT_EQ(clamped(0, 0), 0);  // max(-2, 0)
  EXPECT_EQ(clamped(0, 1), 0);  // max(-1, 0)
  EXPECT_EQ(clamped(0, 2), 0);  // max(0, 0)
  EXPECT_EQ(clamped(1, 0), 1);  // max(1, 0)
  EXPECT_EQ(clamped(1, 2), 3);  // max(3, 0)
}

// Test Max with scalar on left-hand side
TEST(NDArrayFunctionsTest, MaxScalarLeftHandSide) {
  NDArray<int, 2> arr({2, 2});
  arr(0, 0) = -1;
  arr(0, 1) = 5;
  arr(1, 0) = 2;
  arr(1, 1) = -3;

  NDArray<int, 2> res = Max(0, arr);
  EXPECT_EQ(res(0, 0), 0);
  EXPECT_EQ(res(0, 1), 5);
  EXPECT_EQ(res(1, 0), 2);
  EXPECT_EQ(res(1, 1), 0);
}

// Test Max with view and scalar
TEST(NDArrayFunctionsTest, MaxWithView) {
  NDArray<int, 2> base({2, 2});
  base(0, 0) = 1;
  base(0, 1) = 2;
  base(1, 0) = 3;
  base(1, 1) = 4;

  auto v = base.View();

  NDArray<int, 2> mx = Max(v, 3);
  EXPECT_EQ(mx(0, 0), 3);
  EXPECT_EQ(mx(1, 1), 4);
}

// Test shape mismatch throws for Max
TEST(NDArrayFunctionsTest, MaxShapeMismatchThrows) {
  NDArray<int, 2> a({2, 2}, 1);
  NDArray<int, 2> b({2, 3}, 2);
  EXPECT_THROW((void)Max(a, b), std::invalid_argument);
  EXPECT_THROW((void)Min(a, b), std::invalid_argument);
}

// Test composing multiple operations with Max
TEST(NDArrayFunctionsTest, ComposedOperations) {
  NDArray<int, 2> arr({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr(i, j) = static_cast<int>(i * 3 + j + 1);  // 1, 2, 3, 4, 5, 6
    }
  }

  // Compute: Max(-arr, -3)  (clamp negated values to minimum -3)
  auto negated = -arr;
  auto clamped = Max(negated, -3);
  NDArray<int, 2> result = clamped;

  EXPECT_EQ(result(0, 0), -1);  // max(-1, -3)
  EXPECT_EQ(result(0, 1), -2);  // max(-2, -3)
  EXPECT_EQ(result(0, 2), -3);  // max(-3, -3)
  EXPECT_EQ(result(1, 0), -3);  // max(-4, -3)
  EXPECT_EQ(result(1, 2), -3);  // max(-6, -3)
}

// Test scalar operations in complex expressions
TEST(NDArrayFunctionsTest, ScalarInComposedExpression) {
  NDArray<int, 2> arr({2, 2});
  arr(0, 0) = -5;
  arr(0, 1) = 10;
  arr(1, 0) = 3;
  arr(1, 1) = -2;

  // Clamp to [0, 5]: Max(Min(arr, 5), 0)
  NDArray<int, 2> clamped = Max(Min(arr, 5), 0);

  EXPECT_EQ(clamped(0, 0), 0);  // max(min(-5, 5), 0) = 0
  EXPECT_EQ(clamped(0, 1), 5);  // max(min(10, 5), 0) = 5
  EXPECT_EQ(clamped(1, 0), 3);  // max(min(3, 5), 0) = 3
  EXPECT_EQ(clamped(1, 1), 0);  // max(min(-2, 5), 0) = 0
}

// Test deeply composed expressions - this would trigger dangling refs if
// expressions stored operands by reference
TEST(NDArrayFunctionsTest, DeeplyComposedExpressions) {
  NDArray<int, 2> a({2, 2}, 1);
  NDArray<int, 2> b({2, 2}, 2);
  NDArray<int, 2> c({2, 2}, 3);
  NDArray<int, 2> d({2, 2}, 4);

  // Max(a + b, c) - temporary (a + b) expression passed to Max
  NDArray<int, 2> result1 = Max(a + b, c);
  EXPECT_EQ(result1(0, 0), 3);  // max(1+2, 3) = 3

  // (a + b) * (c - d) - two temporary expressions multiplied
  NDArray<int, 2> result2 = (a + b) * (c - d);
  EXPECT_EQ(result2(0, 0), (1 + 2) * (3 - 4));  // 3 * (-1) = -3

  // Max(a, -b) + c - negation temporary in Max, then added to c
  NDArray<int, 2> result3 = Max(a, -b) + c;
  EXPECT_EQ(result3(0, 0), 1 + 3);  // max(1, -2) + 3 = 4

  // Max(Max(a, b), c) - nested Max with temporaries
  NDArray<int, 2> result4 = Max(Max(a, b), c);
  EXPECT_EQ(result4(0, 0), 3);  // max(max(1, 2), 3) = 3
}

// Test complex expression with all operators
TEST(NDArrayFunctionsTest, ComplexComposedExpression) {
  NDArray<int, 2> a({3, 3});
  NDArray<int, 2> b({3, 3});

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      a(i, j) = static_cast<int>(i * 3 + j + 1);     // 1-9
      b(i, j) = static_cast<int>(10 - (i * 3 + j));  // 10-2
    }
  }

  // ((a + b) * Max(-a, 0)) - Min(a * b, 5)
  // This creates a deep expression tree with many temporaries
  auto sum = a + b;
  auto neg_clamped = Max(-a, 0);
  auto product_clamped = Min(a * b, 50);
  NDArray<int, 2> result = (sum * neg_clamped) - product_clamped;

  // Verify at one position: i=0, j=0
  // a(0,0) = 1, b(0,0) = 10
  // sum = 11, neg_clamped = max(-1, 0) = 0
  // product_clamped = min(1*10, 50) = 10
  // result = 11 * 0 - 10 = -10
  EXPECT_EQ(result(0, 0), -10);
}

// Test expression with views in composition
TEST(NDArrayFunctionsTest, ViewsInComposedExpressions) {
  NDArray<int, 2> a({2, 2}, 5);
  NDArray<int, 2> b({2, 2}, 3);

  auto view_a = a.View();
  auto view_b = b.View();

  // Views in composed expression
  NDArray<int, 2> result = (view_a + view_b) * Max(view_a, 4);

  // (5 + 3) * max(5, 4) = 8 * 5 = 40
  EXPECT_EQ(result(0, 0), 40);
  EXPECT_EQ(result(1, 1), 40);
}

// Test that temporary expressions don't cause issues
TEST(NDArrayFunctionsTest, TemporaryExpressionLifetime) {
  NDArray<int, 2> a({2, 2}, 2);
  NDArray<int, 2> b({2, 2}, 3);
  NDArray<int, 2> c({2, 2}, -3);

  // Create result from deeply nested temporary expression
  // If expressions stored refs, this would dangle
  NDArray<int, 2> result = Max(Min(a + b, 10), c);

  // min(2+3, 10) = 5, max(5, -3) = 5
  EXPECT_EQ(result(0, 0), 5);
}

}  // namespace
}  // namespace aifocore::math
