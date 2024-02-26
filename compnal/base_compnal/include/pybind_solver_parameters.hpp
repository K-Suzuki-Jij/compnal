//
//  Copyright 2024 Kohei Suzuki
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
//  pybind_solver_parameters.hpp
//  compnal
//
//  Created by kohei on 2023/06/16.
//  
//

#pragma once

#include "../../../include/solver/parameter_class.hpp"

namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;


void PyBindStateUpdateMethod(py::module &m) {
   py::enum_<compnal::solver::StateUpdateMethod>(m, "StateUpdateMethod")
      .value("METROPOLIS", compnal::solver::StateUpdateMethod::METROPOLIS)
      .value("HEAT_BATH", compnal::solver::StateUpdateMethod::HEAT_BATH)
      .value("SUWA_TODO", compnal::solver::StateUpdateMethod::SUWA_TODO);
}

void PyBindRandomNumberEngine(py::module &m) {
   py::enum_<compnal::solver::RandomNumberEngine>(m, "RandomNumberEngine")
      .value("MT", compnal::solver::RandomNumberEngine::MT)
      .value("MT_64", compnal::solver::RandomNumberEngine::MT_64)
      .value("XORSHIFT", compnal::solver::RandomNumberEngine::XORSHIFT);
}

void SpinSelectionMethod(py::module &m) {
   py::enum_<compnal::solver::SpinSelectionMethod>(m, "SpinSelectionMethod")
      .value("RANDOM", compnal::solver::SpinSelectionMethod::RANDOM)
      .value("SEQUENTIAL", compnal::solver::SpinSelectionMethod::SEQUENTIAL);
}

} // namespace wrapper
} // namespace compnal
