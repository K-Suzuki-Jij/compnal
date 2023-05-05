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
//  pybind_classical_model.hpp
//  compnal
//
//  Created by kohei on 2023/05/03.
//  
//

#ifndef COMPNAL_WRAPPER_CLASSICAL_MODEL_HPP_
#define COMPNAL_WRAPPER_CLASSICAL_MODEL_HPP_

#include "../../../include/model/classical/all.hpp"

namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

template<class LatticeType>
void PyBindClassicalIsing(py::module &m, const std::string &post_name = "") {
   using Ising = model::classical::Ising<LatticeType>;
   std::string name = std::string("Ising") + post_name;
   auto py_class = py::class_<Ising>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&, const double, const double, const double>(), 
                "lattice"_a, "linear"_a, "quadratic"_a, "spin_magnitude"_a);
   
   //Public Member Functions
   py_class.def("get_linear", &Ising::GetLinear);
   py_class.def("get_quadratic", &Ising::GetQuadratic);
   py_class.def("get_twice_spin_magnitude", &Ising::GetTwiceSpinMagnitude);
   py_class.def("calculate_energy", py::overload_cast<const std::vector<typename Ising::PHQType>&>(&Ising::CalculateEnergy, py::const_), "state"_a);
   py_class.def("set_spin_magnitude", &Ising::SetSpinMagnitude, "spin_magnitude"_a, "coordinate"_a);

   m.def("make_ising", [](const LatticeType &lattice,
                          const double linear,
                          const double quadratic,
                          const double spin_magnitude) {
      return model::classical::make_ising<LatticeType>(lattice, linear, quadratic, spin_magnitude);
   }, "lattice"_a, "linear"_a, "quadratic"_a, "spin_magnitude"_a);

}

template<class LatticeType>
void PyBindClassicalPolynomialIsing(py::module &m, const std::string &post_name = "") {
   using PIsing = model::classical::PolynomialIsing<LatticeType>;
   std::string name = std::string("PolynomialIsing") + post_name;
   auto py_class = py::class_<PIsing>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&, const std::unordered_map<std::int32_t, double>&, const double>(), 
                "lattice"_a, "interaction"_a, "spin_magnitude"_a);
   
   //Public Member Functions
   py_class.def("get_interaction", &PIsing::GetInteraction);
   py_class.def("calculate_energy", py::overload_cast<const std::vector<typename PIsing::PHQType>&>(&PIsing::CalculateEnergy, py::const_), "state"_a);
   py_class.def("get_degree", &PIsing::GetDegree);
   py_class.def("get_twice_spin_magnitude", &PIsing::GetTwiceSpinMagnitude);
   py_class.def("set_spin_magnitude", &PIsing::SetSpinMagnitude, "spin_magnitude"_a, "coordinate"_a);

   m.def("make_polynomial_ising", [](const LatticeType &lattice,
                                     const std::unordered_map<std::int32_t, double> &interaction,
                                     const double spin_magnitude) {
      return model::classical::make_polynomial_ising<LatticeType>(lattice, interaction, spin_magnitude);
   }, "lattice"_a, "interaction"_a, "spin_magnitude"_a);

}


} // namespace wrapper
} // namespace compnal

#endif /* COMPNAL_WRAPPER_CLASSICAL_MODEL_HPP_ */
