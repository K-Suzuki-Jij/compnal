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
//  pybind_classical_monte_carlo.hpp
//  compnal
//
//  Created by kohei on 2023/06/16.
//  
//

#pragma once

#include "../../../include/solver/classical_monte_carlo/classical_monte_carlo.hpp"

namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;


template<class ModelType>
void PyBindClassicalMonteCarlo(py::module &m, const std::string &post_name = "") {
   using CMC = solver::classical_monte_carlo::ClassicalMonteCarlo<ModelType>;
   std::string name = std::string("ClassicalMonteCarlo") + post_name;
   auto py_class = py::class_<CMC>(m, name.c_str(), py::module_local());

   //Constructors
   py_class.def(py::init<>());

   py_class.def("run_single_flip", &CMC::RunSingleFlip, 
                "model"_a, "num_sweeps"_a, "num_samples"_a, "num_threads"_a, "temperature"_a,
                "seed"_a, "updater"_a, "random_number_engine"_a, "spin_selector"_a);
   py_class.def("run_parallel_tempering", &CMC::RunParallelTempering, 
                "model"_a, "num_sweeps"_a, "num_swaps"_a, "num_samples"_a, "num_threads"_a, 
                "temperature_list"_a, "seed"_a, "updater"_a, 
                "random_number_engine"_a, "spin_selector"_a);
   py_class.def("run_multi_flip", &CMC::RunMultiFlip,
                "model"_a, "num_update_variables"_a, "num_sweeps"_a, "num_samples"_a, "num_threads"_a, "temperature"_a,
                "seed"_a, "updater"_a, "random_number_engine"_a, "spin_selector"_a);
   py_class.def("calculate_energies", &CMC::CalculateEnergies, 
                "model"_a, "samples"_a, "num_threads"_a);

   m.def("make_classical_monte_carlo", [](const ModelType &model) {
      return solver::classical_monte_carlo::make_classical_monte_carlo<ModelType>();
   });
}


} // namespace wrapper
} // namespace compnal