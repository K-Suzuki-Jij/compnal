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
//  Created by Kohei Suzuki on 2022/07/13.
//

#ifndef COMPNAL_PYBIND11_MODEL_HPP_
#define COMPNAL_PYBIND11_MODEL_HPP_

#include "../../../cpp_compnal/src/model/all.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>


namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

template<class LatticeType, typename RealType>
void pybind11ModelIsing(py::module &m, const std::string &post_name = "") {
   
   using Ising = model::Ising<LatticeType, RealType>;
   std::string name = std::string("Ising") + post_name;
   
   auto py_class = py::class_<Ising>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&, const typename Ising::LinearType&, const typename Ising::QuadraticType&>(), "lattice"_a, "interaction_deg_1"_a, "interaction_deg_2"_a);
   
   //Public Member Functions
   py_class.def("get_system_size", &Ising::GetSystemSize);
   py_class.def("get_boundary_condition", &Ising::GetBoundaryCondition);
   py_class.def("get_degree", &Ising::GetDegree);
   py_class.def("generate_index_list", &Ising::GenerateIndexList);
   py_class.def("calculate_energy", py::overload_cast<const std::vector<typename Ising::OPType>&>(&Ising::CalculateEnergy, py::const_), "sample"_a);
 
   m.def("make_ising", [](const LatticeType &lattice,
                          const typename Ising::LinearType &linear,
                          const typename Ising::QuadraticType &quadratic) {
      return model::make_ising<LatticeType, RealType>(lattice, linear, quadratic);
   }, "lattice"_a, "linear"_a, "quadratic"_a);
}

template<class LatticeType, typename RealType>
void pybind11ModelPolynomialIsing(py::module &m, const std::string &post_name = "") {
   
   using PolyIsing = model::PolynomialIsing<LatticeType, RealType>;
   std::string name = std::string("PolynomialIsing") + post_name;

   auto py_class = py::class_<PolyIsing>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&, const typename PolyIsing::PolynomialType&>(), "lattice"_a, "interaction"_a);
   
   //Public Member Functions
   py_class.def("get_system_size", &PolyIsing::GetSystemSize);
   py_class.def("get_boundary_condition", &PolyIsing::GetBoundaryCondition);
   py_class.def("get_degree", &PolyIsing::GetDegree);
   py_class.def("generate_index_list", &PolyIsing::GenerateIndexList);
   py_class.def("calculate_energy", py::overload_cast<const std::vector<typename PolyIsing::OPType>&>(&PolyIsing::CalculateEnergy, py::const_), "sample"_a);
   
   m.def("make_polynomial_ising", [](const LatticeType &lattice, const typename PolyIsing::PolynomialType &interaction) {
      return model::make_polynomial_ising<LatticeType, RealType>(lattice, interaction);
   }, "lattice"_a, "interaction"_a);
}

template<class LatticeType, typename RealType>
void pybind11ModelHubbard(py::module &m, const std::string &post_name = "") {
   
   using Hub = model::quantum::Hubbard<LatticeType, RealType>;
   std::string name = std::string("Hubbard") + post_name;

   auto py_class = py::class_<Hub>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&>(), "lattice"_a);
   
   //Public Member Functions
   py_class.def("set_total_electron", &Hub::SetTotalElectron, "total_electron"_a);
   py_class.def("set_total_sz"      , &Hub::SetTotalSz      , "total_sz"_a      );
   
   py_class.def("set_hopping_energy"    , &Hub::SetHoppingEnergy    , "hopping_energy"_a    );
   py_class.def("set_onsite_coulomb"    , &Hub::SetOnsiteCoulomb    , "onsite_coulomb"_a    );
   py_class.def("set_intersite_coulomb" , &Hub::SetIntersiteCoulomb , "intersite_coulomb"_a );
   py_class.def("set_chemical_potential", &Hub::SetChemicalPotential, "chemical_potential"_a);
   py_class.def("set_magnetic_field"    , &Hub::SetMagneticField    , "magnetic_field"_a    );
   
   py_class.def("get_hopping_energy"    , &Hub::GetHoppingEnergy    );
   py_class.def("get_onsite_coulomb"    , &Hub::GetOnsiteCoulomb    );
   py_class.def("get_intersite_coulomb" , &Hub::GetIntersiteCoulomb );
   py_class.def("get_chemical_potential", &Hub::GetChemicalPotential);
   py_class.def("get_magnetic_field"    , &Hub::GetMagneticField    );

   py_class.def("get_onsite_operator_CUp", &Hub::GetOnsiteOperatorCUp);
   py_class.def("get_onsite_operator_CDown", &Hub::GetOnsiteOperatorCDown);

   py_class.def("get_onsite_operator_CUpDagger", &Hub::GetOnsiteOperatorCUpDagger);
   py_class.def("get_onsite_operator_CDownDagger", &Hub::GetOnsiteOperatorCDownDagger);

   py_class.def("get_onsite_operator_NCUp", &Hub::GetOnsiteOperatorNCUp);
   py_class.def("get_onsite_operator_NCDown", &Hub::GetOnsiteOperatorNCDown);

   py_class.def("get_onsite_operator_NC", &Hub::GetOnsiteOperatorNC);
   py_class.def("get_onsite_operator_Sx", &Hub::GetOnsiteOperatorSx);
   py_class.def("get_onsite_operator_iSy", &Hub::GetOnsiteOperatoriSy);
   py_class.def("get_onsite_operator_Sz", &Hub::GetOnsiteOperatorSz);
   py_class.def("get_onsite_operator_Sp", &Hub::GetOnsiteOperatorSp);
   py_class.def("get_onsite_operator_Sm", &Hub::GetOnsiteOperatorSm);
   
}

template<class LatticeType, typename RealType>
void pybind11ModelHeisenberg(py::module &m, const std::string &post_name = "") {
   
   using Hei = model::quantum::Heisenberg<LatticeType, RealType>;
   std::string name = std::string("Heisenberg") + post_name;

   auto py_class = py::class_<Hei>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&>(), "lattice"_a);
   
   //Public Member Functions
   py_class.def("set_total_sz", &Hei::SetTotalSz, "total_sz"_a);
   py_class.def("set_magnitude_spin", &Hei::SetMagnitudeSpin, "magnitude_spin"_a);
   
   py_class.def("set_spin_spin_z"    , &Hei::SetSpinSpinZ    , "spin_spin_z"_a   );
   py_class.def("set_spin_spin_xy"    , &Hei::SetSpinSpinXY   , "spin_spin_xy"_a  );
   py_class.def("set_magnetic_field" , &Hei::SetMagneticField, "magnetic_field"_a);
   py_class.def("set_anisotropy", &Hei::SetAnisotropy   , "anisotropy"_a    );
   
   py_class.def("get_spin_spin_z"    , &Hei::GetSpinSpinZ    );
   py_class.def("get_spin_spin_xy"    , &Hei::GetSpinSpinXY   );
   py_class.def("get_magnetic_field" , &Hei::GetMagneticField);
   py_class.def("get_anisotropy", &Hei::GetAnisotropy   );

   py_class.def("get_magnitude_spin", &Hei::GetMagnitudeSpin);
   py_class.def("get_onsite_operator_Sx" , &Hei::GetOnsiteOperatorSx );
   py_class.def("get_onsite_operator_iSy", &Hei::GetOnsiteOperatoriSy);
   py_class.def("get_onsite_operator_Sz" , &Hei::GetOnsiteOperatorSz );
   py_class.def("get_onsite_operator_Sp" , &Hei::GetOnsiteOperatorSp );
   py_class.def("get_onsite_operator_Sm" , &Hei::GetOnsiteOperatorSm );

}


template<class LatticeType, typename RealType>
void pybind11ModelKondoLattice(py::module &m, const std::string &post_name = "") {
   
   using KL = model::quantum::KondoLattice<LatticeType, RealType>;
   std::string name = std::string("KondoLattice") + post_name;

   auto py_class = py::class_<KL>(m, name.c_str(), py::module_local());
   
   //Constructors
   py_class.def(py::init<const LatticeType&>(), "lattice"_a);
   
   //Public Member Functions
   py_class.def("set_total_sz", &KL::SetTotalSz, "total_sz"_a);
   py_class.def("set_total_electron", &KL::SetTotalElectron, "total_electron"_a);
   py_class.def("set_magnitude_spin", &KL::SetMagnitudeLSpin, "magnitude_spin"_a);
   
   py_class.def("set_hopping_energy"    , &KL::SetHoppingEnergy    , "hopping_energy"_a    );
   py_class.def("set_onsite_coulomb"    , &KL::SetOnsiteCoulomb    , "onsite_coulomb"_a    );
   py_class.def("set_intersite_coulomb" , &KL::SetIntersiteCoulomb , "intersite_coulomb"_a );
   py_class.def("set_chemical_potential", &KL::SetChemicalPotential, "chemical_potential"_a);
   py_class.def("set_magnetic_field_electron", &KL::SetMagneticFieldElectron, "magnetic_field_electron"_a    );

   py_class.def("set_kondo_exchange_z"    , &KL::SetKondoExchangeZ    , "kondo_exchange_z"_a   );
   py_class.def("set_kondo_exchange_xy"    , &KL::SetKondoExchangeXY   , "kondo_exchange_xy"_a  );
   py_class.def("set_magnetic_field_spin" , &KL::SetMagneticFieldSpin, "magnetic_field_spin"_a);
   py_class.def("set_anisotropy", &KL::SetAnisotropy   , "anisotropy"_a    );

   py_class.def("get_magnitude_spin", &KL::GetMagnitudeLSpin);

   
   py_class.def("get_hopping_energy"    , &KL::GetHoppingEnergy    );
   py_class.def("get_onsite_coulomb"    , &KL::GetOnsiteCoulomb    );
   py_class.def("get_intersite_coulomb" , &KL::GetIntersiteCoulomb );
   py_class.def("get_chemical_potential", &KL::GetChemicalPotential);
   py_class.def("get_magnetic_field_electron", &KL::GetMagneticFieldElectron);

   py_class.def("get_kondo_exchange_z"    , &KL::GetKondoExchangeZ    );
   py_class.def("get_kondo_exchange_xy"    , &KL::GetKondoExchangeXY   );
   py_class.def("get_magnetic_field_spin" , &KL::GetMagneticFieldSpin);
   py_class.def("get_anisotropy", &KL::GetAnisotropy   );

   py_class.def("get_onsite_operator_CUp", &KL::GetOnsiteOperatorCUp);
   py_class.def("get_onsite_operator_CDown", &KL::GetOnsiteOperatorCDown);

   py_class.def("get_onsite_operator_CUpDagger", &KL::GetOnsiteOperatorCUpDagger);
   py_class.def("get_onsite_operator_CDownDagger", &KL::GetOnsiteOperatorCDownDagger);

   py_class.def("get_onsite_operator_NCUp", &KL::GetOnsiteOperatorNCUp);
   py_class.def("get_onsite_operator_NCDown", &KL::GetOnsiteOperatorNCDown);

   py_class.def("get_onsite_operator_NC", &KL::GetOnsiteOperatorNC);
   py_class.def("get_onsite_operator_SxC", &KL::GetOnsiteOperatorSxC);
   py_class.def("get_onsite_operator_iSyC", &KL::GetOnsiteOperatoriSyC);
   py_class.def("get_onsite_operator_SzC", &KL::GetOnsiteOperatorSzC);
   py_class.def("get_onsite_operator_SpC", &KL::GetOnsiteOperatorSpC);
   py_class.def("get_onsite_operator_SmC", &KL::GetOnsiteOperatorSmC);

   py_class.def("get_onsite_operator_SxL" , &KL::GetOnsiteOperatorSxL );
   py_class.def("get_onsite_operator_iSyL", &KL::GetOnsiteOperatoriSyL);
   py_class.def("get_onsite_operator_SzL" , &KL::GetOnsiteOperatorSzL );
   py_class.def("get_onsite_operator_SpL" , &KL::GetOnsiteOperatorSpL );
   py_class.def("get_onsite_operator_SmL" , &KL::GetOnsiteOperatorSmL );

}

} // namespace wrapper
} // namespace compnal



#endif /* COMPNAL_PYBIND11_MODEL_HPP_ */
