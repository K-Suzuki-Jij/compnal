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

#ifndef COMPNAL_WRAPPER_CLASSICAL_MONTE_CARLO_HPP_
#define COMPNAL_WRAPPER_CLASSICAL_MONTE_CARLO_HPP_

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
   py_class.def(py::init<const ModelType&>(), "model"_a);

   //Public Member Functions
   py_class.def("set_num_sweeps", &CMC::SetNumSweeps, "num_sweeps"_a);
   py_class.def("set_num_samples", &CMC::SetNumSamples, "num_samples"_a);
   py_class.def("set_num_threads", &CMC::SetNumThreads, "num_threads"_a);
   py_class.def("set_temperature", &CMC::SetTemperature, "temperature"_a);
   py_class.def("set_state_update_method", &CMC::SetStateUpdateMethod, "state_update_method"_a);
   py_class.def("set_random_number_engine", &CMC::SetRandomNumberEngine, "random_number_engine"_a);
   py_class.def("set_spin_selection_method", &CMC::SetSpinSelectionMethod, "spin_selection_method"_a);
   
   py_class.def("get_num_sweeps", &CMC::GetNumSweeps);
   py_class.def("get_num_samples", &CMC::GetNumSamples);
   py_class.def("get_num_threads", &CMC::GetNumThreads);
   py_class.def("get_temperature", &CMC::GetTemperature);
   py_class.def("get_state_update_method", &CMC::GetStateUpdateMethod);
   py_class.def("get_random_number_engine", &CMC::GetRandomNumberEngine);
   py_class.def("get_spin_selection_method", &CMC::GetSpinSelectionMethod);
   py_class.def("get_seed", &CMC::GetSeed);
   py_class.def("get_samples", &CMC::GetSamples);
   py_class.def("get_model", &CMC::GetModel);
   py_class.def("calculate_energies", &CMC::CalculateEnergies);
   py_class.def("run_sampling", py::overload_cast<>(&CMC::RunSampling));
   py_class.def("run_sampling", py::overload_cast<const std::uint64_t>(&CMC::RunSampling), "seed"_a);

   m.def("make_classical_monte_carlo", [](const ModelType &model) {
      return solver::classical_monte_carlo::make_classical_monte_carlo<ModelType>(model);
   }, "model"_a);
}


} // namespace wrapper
} // namespace compnal


#endif /* COMPNAL_WRAPPER_CLASSICAL_MONTE_CARLO_HPP_ */