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
//  pybind_utility.hpp
//  compnal
//
//  Created by kohei on 2023/07/25.
//  
//

#pragma once

#include "../../../include/utility/all.hpp"

namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

void PyBindUtility(py::module &m) {
    m.def("calculate_moment", &utility::CalculateMoment, "samples"_a, "order"_a, "bias"_a, "num_threads"_a);
    m.def("calculate_moment_with_std", &utility::CalculateMomentWithSTD, "samples"_a, "order"_a, "bias"_a, "num_threads"_a);
}

} // namespace wrapper
} // namespace compnal
