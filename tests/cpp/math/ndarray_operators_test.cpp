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

// Test element-wise addition
TEST(NDArrayOperatorsTest, ElementWiseAdd) {
  NDArray<int, 2> arr1({2, 2}, 5);
  NDArray<int, 2> arr2({2, 2}, 3);

  NDArray<int, 2> result = arr1 + arr2;

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(result(i, j), 8);
    }
  }
}

// Test element-wise subtraction
TEST(NDArrayOperatorsTest, ElementWiseSub) {
  NDArray<int, 2> arr1({2, 2}, 10);
  NDArray<int, 2> arr2({2, 2}, 3);

  NDArray<int, 2> result = arr1 - arr2;

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(result(i, j), 7);
    }
  }
}

// Test element-wise multiplication
TEST(NDArrayOperatorsTest, ElementWiseMul) {
  NDArray<int, 2> arr1({2, 2}, 4);
  NDArray<int, 2> arr2({2, 2}, 3);

  NDArray<int, 2> result = arr1 * arr2;

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(result(i, j), 12);
    }
  }
}

// Test shape mismatch throws for operators
TEST(NDArrayOperatorsTest, BinaryShapeMismatchThrows) {
  NDArray<int, 2> a({2, 2}, 1);
  NDArray<int, 2> b({2, 3}, 2);
  EXPECT_THROW((void)(a + b), std::invalid_argument);
  EXPECT_THROW((void)(a - b), std::invalid_argument);
  EXPECT_THROW((void)(a * b), std::invalid_argument);
}

// Test nested expressions with operators
TEST(NDArrayOperatorsTest, NestedExpressions) {
  NDArray<int, 2> a({2, 2}, 1);
  NDArray<int, 2> b({2, 2}, 2);

  // (a + b) * (b - a)
  NDArray<int, 2> res = (a + b) * (b - a);
  EXPECT_EQ(res(0, 0), (1 + 2) * (2 - 1));  // 3
}

}  // namespace
}  // namespace aifocore::math
