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
//  ising_cubic.hpp
//  compnal
//
//  Created by kohei on 2023/06/13.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CUBIC_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CUBIC_HPP_

#include "template_system.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/cubic.hpp"
#include "../../../lattice/boundary_condition.hpp"


namespace compnal {
namespace solver {
namespace classical_monte_carlo {


//! @brief System class for the Ising model on a cubic.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::Ising<lattice::Cubic>, RandType>: public BaseIsingSystem<model::classical::Ising<lattice::Cubic>, RandType> {
   //! @brief Model type.
   using ModelType = model::classical::Ising<lattice::Cubic>;
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   linear_(model.GetLinear()),
   quadratic_(model.GetQuadratic()),
   x_size_(model.GetLattice().GetXSize()),
   y_size_(model.GetLattice().GetYSize()),
   z_size_(model.GetLattice().GetZSize()){
      this->d_E_ = GenerateEnergyDifference(this->sample_);
   }
   
   //! @brief Set sample by states.
   //! Here, the states represents energy levels. For example for S=1/2 ising spins,
   //! s=-1/2 corresponds to the state being 0 and s=1/2 corresponds to the state being 1.
   //! @param state_list The list of states.
   void SetSampleByState(const std::vector<std::int32_t> &state_list) {
      if (state_list.size() != this->sample_.size()) {
         throw std::invalid_argument("The size of initial variables is not equal to the system size.");
      }
      for (std::size_t i = 0; i < this->sample_.size(); ++i) {
         this->sample_[i].SetState(state_list[i]);
      }
      this->d_E_ = GenerateEnergyDifference(this->sample_);
      this->energy_ = this->model_.CalculateEnergy(this->ExtractSample());
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
      this->energy_ += this->GetEnergyDifference(index, update_state);
      const double diff = this->quadratic_*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
      const std::int32_t coo_x = index%x_size_;
      const std::int32_t coo_y = (index%(x_size_*y_size_))/x_size_;
      const std::int32_t coo_z = index/(x_size_*y_size_);
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x + 1)%x_size_] += diff;
         this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - 1 + x_size_)%x_size_] += diff;
         this->d_E_[coo_z*x_size_*y_size_ + ((coo_y + 1)%y_size_)*x_size_ + coo_x] += diff;
         this->d_E_[coo_z*x_size_*y_size_ + ((coo_y - 1 + y_size_)%y_size_)*x_size_ + coo_x] += diff;
         this->d_E_[((coo_z + 1)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += diff;
         this->d_E_[((coo_z - 1 + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += diff;
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         // x-direction
         if (coo_x < x_size_ - 1) {
            this->d_E_[index + 1] += diff;
         }
         if (coo_x > 0) {
            this->d_E_[index - 1] += diff;
         }
         
         // y-direction
         if (coo_y < y_size_ - 1) {
            this->d_E_[coo_z*x_size_*y_size_ + (coo_y + 1)*x_size_ + coo_x] += diff;
         }
         if (coo_y > 0) {
            this->d_E_[coo_z*x_size_*y_size_ + (coo_y - 1)*x_size_ + coo_x] += diff;
         }
         
         // z-direction
         if (coo_z < z_size_ - 1) {
            this->d_E_[(coo_z + 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += diff;
         }
         if (coo_z > 0) {
            this->d_E_[(coo_z - 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += diff;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
   
private:
   //! @brief The linear interaction.
   const double linear_ = 0;

   //! @brief The quadratic interaction.
   const double quadratic_ = 0;
   
   //! @brief The length of x-direction.
   const std::int32_t x_size_ = 0;

   //! @brief The length of y-direction.
   const std::int32_t y_size_ = 0;
   
   //! @brief The length of z-direction.
   const std::int32_t z_size_ = 0;
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const std::vector<model::utility::Spin> &sample) const {
      std::vector<double> d_E_(this->system_size_);
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size_; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size_; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size_; ++coo_z) {
                  const std::int32_t index_xp1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x + 1)%x_size_;
                  const std::int32_t index_xm1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - 1 + x_size_)%x_size_;
                  const std::int32_t index_yp1 = coo_z*x_size_*y_size_ + ((coo_y + 1)%y_size_)*x_size_ + coo_x;
                  const std::int32_t index_ym1 = coo_z*x_size_*y_size_ + ((coo_y - 1 + y_size_)%y_size_)*x_size_ + coo_x;
                  const std::int32_t index_zp1 = ((coo_z + 1)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                  const std::int32_t index_zm1 = ((coo_z - 1 + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                  
                  const auto v_xp1 = sample[index_xp1].GetValue();
                  const auto v_xm1 = sample[index_xm1].GetValue();
                  const auto v_yp1 = sample[index_yp1].GetValue();
                  const auto v_ym1 = sample[index_ym1].GetValue();
                  const auto v_zp1 = sample[index_zp1].GetValue();
                  const auto v_zm1 = sample[index_zm1].GetValue();
                  d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x] += this->quadratic_*(v_xp1 + v_xm1 + v_yp1 + v_ym1 + v_zp1 + v_zm1) + this->linear_;
               }
            }
         }
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size_; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size_; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size_; ++coo_z) {
                  const std::int32_t index = coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                  if (coo_x < x_size_ - 1) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x + 1].GetValue();
                  }
                  if (coo_x > 0) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x - 1].GetValue();
                  }
                  if (coo_y < y_size_ - 1) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size_*y_size_ + (coo_y + 1)*x_size_ + coo_x].GetValue();
                  }
                  if (coo_y > 0) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size_*y_size_ + (coo_y - 1)*x_size_ + coo_x].GetValue();
                  }
                  if (coo_z < z_size_ - 1) {
                     d_E_[index] += this->quadratic_*sample[(coo_z + 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x].GetValue();
                  }
                  if (coo_z > 0) {
                     d_E_[index] += this->quadratic_*sample[(coo_z - 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x].GetValue();
                  }
                  d_E_[index] += this->linear_;
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      return d_E_;
      
   }
   
};


} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CUBIC_HPP_ */
