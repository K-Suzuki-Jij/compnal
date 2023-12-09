//
//  Copyright 2023 Kohei Suzuki
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
//  test_variable.hpp
//  compnal
//
//  Created by kohei on 2023/06/02.
//  
//

#pragma once

#include "../../../../include/model/utility/variable.hpp"
#include "../../../../include/utility/random.hpp"

namespace compnal {
namespace test {

TEST(ModelUtility, Spin) {
   EXPECT_THROW((model::utility::Spin{0.3, 1}), std::invalid_argument);
   EXPECT_THROW((model::utility::Spin{0.5, 0}), std::invalid_argument);
   
   model::utility::Spin spin{0.5, 2};
   EXPECT_THROW(spin.GetValueFromState(2), std::invalid_argument);
   EXPECT_THROW(spin.GetValueFromState(-1), std::invalid_argument);
   EXPECT_DOUBLE_EQ(spin.GetValueFromState(0), -1);
   EXPECT_DOUBLE_EQ(spin.GetValueFromState(1), +1);
   EXPECT_DOUBLE_EQ(spin.GetValue(), -1);
   EXPECT_EQ(spin.GetStateNumber(), 0);
   
   std::mt19937 engine(0);
   EXPECT_EQ(spin.GenerateCandidateState(&engine), 1);
   EXPECT_NO_THROW(spin.SetState(1));
   EXPECT_EQ(spin.GenerateCandidateState(&engine), 0);
   EXPECT_THROW(spin.SetState(-1), std::invalid_argument);
   EXPECT_THROW(spin.SetState(2), std::invalid_argument);
   
   std::mt19937_64 engine_64(0);
   EXPECT_NO_THROW(spin.SetState(0));
   EXPECT_EQ(spin.GenerateCandidateState(&engine_64), 1);
   EXPECT_NO_THROW(spin.SetState(1));
   EXPECT_EQ(spin.GenerateCandidateState(&engine_64), 0);
   EXPECT_THROW(spin.SetState(-1), std::invalid_argument);
   EXPECT_THROW(spin.SetState(2), std::invalid_argument);
   
   utility::Xorshift xorshift(0);
   EXPECT_NO_THROW(spin.SetState(0));
   EXPECT_EQ(spin.GenerateCandidateState(&xorshift), 1);
   EXPECT_NO_THROW(spin.SetState(1));
   EXPECT_EQ(spin.GenerateCandidateState(&xorshift), 0);
   EXPECT_THROW(spin.SetState(-1), std::invalid_argument);
   EXPECT_THROW(spin.SetState(2), std::invalid_argument);
   
}

}  // namespace test
}  // namespace compnal

