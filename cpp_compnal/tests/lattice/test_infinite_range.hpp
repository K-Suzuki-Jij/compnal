//
//  Copyright 2022 Kohei Suzuki
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  test_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_TEST_LATTICE_INFINITE_RANGE_HPP_
#define COMPNAL_TEST_LATTICE_INFINITE_RANGE_HPP_

#include "../../src/lattice/infinite_range.hpp"
#include "../../src/lattice/cubic.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(LatticeInfiniteRange, Constructor) {
   EXPECT_EQ(lattice::InfiniteRange{2}.GetSystemSize(), 2);
   EXPECT_EQ(lattice::InfiniteRange{2}.GetBoundaryCondition(), lattice::BoundaryCondition::NONE);
   EXPECT_THROW(lattice::InfiniteRange{-1}.GetSystemSize(), std::runtime_error);
}


} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_LATTICE_INFINITE_RANGE_HPP_ */
