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
//  test_electron.hpp
//  compnal
//
//  Created by kohei on 2022/12/31.
//  
//

#ifndef COMPNAL_TEST_QUANTUM_ELECTRON_HPP_
#define COMPNAL_TEST_QUANTUM_ELECTRON_HPP_

#include "../../../src/model/quantum/electron.hpp"

namespace compnal {
namespace test {

TEST(ModelQuantum, Electron) {
   lattice::Chain chain{10};
   model::quantum::Electron<lattice::Chain, double>(chain, 10, 0);
   

}

TEST(ModelQuantum, ElectronValidateQNumber) {
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 4, 0   }.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 3, +0.5}.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 3, -0.5}.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 2, 0   }.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 2, +1  }.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 2, -1  }.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 1, +0.5}.ValidateQNumber()));
   EXPECT_TRUE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, 1, -0.5}.ValidateQNumber()));

   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +4, +1  }.ValidateQNumber()));
   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +3, +0  }.ValidateQNumber()));
   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +3, -1.5}.ValidateQNumber()));
   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +2, +0.5}.ValidateQNumber()));
   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +2, -0.5}.ValidateQNumber()));
   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +1, +1  }.ValidateQNumber()));
   EXPECT_FALSE((model::quantum::Electron<lattice::Chain, double>{lattice::Chain{2}, +1, -1  }.ValidateQNumber()));
}


} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_QUANTUM_ELECTRON_HPP_ */
