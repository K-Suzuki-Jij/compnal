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
//  Created by Kohei Suzuki on 2021/06/14.
//


#include "utility/test_integer.hpp"
#include "lattice/test_base_one_dimensional_lattice.hpp"
#include "lattice/test_base_two_dimensional_lattice.hpp"
#include "lattice/test_base_three_dimensional_lattice.hpp"
#include "lattice/test_infinite_range.hpp"
#include "lattice/test_any_lattice.hpp"
#include "interaction/test_quadratic_any_interaction.hpp"
#include "interaction/test_polynomial_any_interaction.hpp"
#include "model/classical/test_polynomial_ising.hpp"
#include "model/classical/test_ising.hpp"
#include "model/quantum/test_electron.hpp"
#include "solver/test_cmc_polynomial_ising.hpp"
#include "solver/test_cmc_ising.hpp"
#include "blas/test_crs.hpp"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
