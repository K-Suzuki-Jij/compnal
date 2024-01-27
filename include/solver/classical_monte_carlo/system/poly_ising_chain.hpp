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
//  poly_ising_chain.hpp
//  compnal
//
//  Created by kohei on 2023/06/25.
//  
//

#pragma once

#include "template_system.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/chain.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief System class for the Ising model on a chain.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::PolynomialIsing<lattice::Chain>, RandType>: public BaseIsingSystem<model::classical::PolynomialIsing<lattice::Chain>, RandType> {
  
   //! @brief Model type.
   using ModelType = model::classical::PolynomialIsing<lattice::Chain>;
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   interaction_(model.GetInteraction()),
   degree_(model.GetDegree()) {
      this->d_E_ = model.GenerateEnergyDifference(this->sample_);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
      this->energy_ += this->GetEnergyDifference(index, update_state);
      if (degree_ == 2) {
         Flip2Body(index, update_state);
      }
      else if (degree_ == 3) {
         Flip3Body(index, update_state);
      }
      else if (degree_ == 4) {
         Flip4Body(index, update_state);
      }
      else if (degree_ == 5) {
         Flip5Body(index, update_state);
      }
      else {
         FlipAny(index, update_state);
      }
   }
   
private:
   //! @brief The polynomial interaction.
   std::unordered_map<std::int32_t, double> interaction_;
   
   //! @brief The degree of the interactions.
   std::int32_t degree_ = 0;
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip2Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = interaction_.at(2)*diff;
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         const std::int32_t m1 = (index - 1 + this->system_size_)%this->system_size_;
         const std::int32_t p1 = (index + 1)%this->system_size_;
         this->d_E_[m1] += val_2;
         this->d_E_[p1] += val_2;
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         const std::int32_t m1 = index - 1;
         const std::int32_t p1 = index + 1;
         if (m1 >= 0) {
            this->d_E_[m1] += val_2;
         }
         if (p1 < this->system_size_) {
            this->d_E_[p1] += val_2;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip3Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = (interaction_.count(2) == 1 ? interaction_.at(2) : 0)*diff;
      const double val_3 = interaction_.at(3)*diff;
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         const std::int32_t m2 = (index - 2 + this->system_size_)%this->system_size_;
         const std::int32_t m1 = (index - 1 + this->system_size_)%this->system_size_;
         const std::int32_t p1 = (index + 1)%this->system_size_;
         const std::int32_t p2 = (index + 2)%this->system_size_;
         this->d_E_[m2] += val_3*this->sample_[m1].GetValue();
         this->d_E_[m1] += val_3*(this->sample_[m2].GetValue() + this->sample_[p1].GetValue()) + val_2;
         this->d_E_[p1] += val_3*(this->sample_[m1].GetValue() + this->sample_[p2].GetValue()) + val_2;
         this->d_E_[p2] += val_3*this->sample_[p1].GetValue();
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         const std::int32_t m2 = index - 2;
         const std::int32_t m1 = index - 1;
         const std::int32_t p1 = index + 1;
         const std::int32_t p2 = index + 2;
         const double s_m2 = m2 >= 0 ? this->sample_[m2].GetValue() : 0;
         const double s_m1 = m1 >= 0 ? this->sample_[m1].GetValue() : 0;
         const double s_p1 = p1 < this->system_size_ ? this->sample_[p1].GetValue() : 0;
         const double s_p2 = p2 < this->system_size_ ? this->sample_[p2].GetValue() : 0;
         if (m2 >= 0) {
            this->d_E_[m2] += val_3*s_m1;
         }
         if (m1 >= 0) {
            this->d_E_[m1] += val_3*(s_m2 + s_p1) + val_2;
         }
         if (p1 < this->system_size_) {
            this->d_E_[p1] += val_3*(s_m1 + s_p2) + val_2;
         }
         if (p2 < this->system_size_) {
            this->d_E_[p2] += val_3*s_p1;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip4Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = (interaction_.count(2) == 1 ? interaction_.at(2) : 0)*diff;
      const double val_3 = (interaction_.count(3) == 1 ? interaction_.at(3) : 0)*diff;
      const double val_4 = interaction_.at(4)*diff;
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         const std::int32_t m3 = (index - 3 + this->system_size_)%this->system_size_;
         const std::int32_t m2 = (index - 2 + this->system_size_)%this->system_size_;
         const std::int32_t m1 = (index - 1 + this->system_size_)%this->system_size_;
         const std::int32_t p1 = (index + 1)%this->system_size_;
         const std::int32_t p2 = (index + 2)%this->system_size_;
         const std::int32_t p3 = (index + 3)%this->system_size_;
         const double s_m3 = this->sample_[m3].GetValue();
         const double s_m2 = this->sample_[m2].GetValue();
         const double s_m1 = this->sample_[m1].GetValue();
         const double s_p1 = this->sample_[p1].GetValue();
         const double s_p2 = this->sample_[p2].GetValue();
         const double s_p3 = this->sample_[p3].GetValue();
         this->d_E_[m3] += val_4*s_m2*s_m1;
         this->d_E_[m2] += val_4*(s_m3*s_m1 + s_m1*s_p1) + val_3*s_m1;
         this->d_E_[m1] += val_4*(s_m3*s_m2 + s_m2*s_p1 + s_p1*s_p2) + val_3*(s_m2 + s_p1) + val_2;
         this->d_E_[p1] += val_4*(s_m2*s_m1 + s_m1*s_p2 + s_p2*s_p3) + val_3*(s_m1 + s_p2) + val_2;
         this->d_E_[p2] += val_4*(s_m1*s_p1 + s_p1*s_p3) + val_3*s_p1;
         this->d_E_[p3] += val_4*s_p1*s_p2;
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         const std::int32_t m3 = index - 3;
         const std::int32_t m2 = index - 2;
         const std::int32_t m1 = index - 1;
         const std::int32_t p1 = index + 1;
         const std::int32_t p2 = index + 2;
         const std::int32_t p3 = index + 3;
         const double s_m3 = m3 >= 0 ? this->sample_[m3].GetValue() : 0;
         const double s_m2 = m2 >= 0 ? this->sample_[m2].GetValue() : 0;
         const double s_m1 = m1 >= 0 ? this->sample_[m1].GetValue() : 0;
         const double s_p1 = p1 < this->system_size_ ? this->sample_[p1].GetValue() : 0;
         const double s_p2 = p2 < this->system_size_ ? this->sample_[p2].GetValue() : 0;
         const double s_p3 = p3 < this->system_size_ ? this->sample_[p3].GetValue() : 0;
         if (m3 >= 0) {
            this->d_E_[m3] += val_4*s_m2*s_m1;
         }
         if (m2 >= 0) {
            this->d_E_[m2] += val_4*(s_m3*s_m1 + s_m1*s_p1) + val_3*s_m1;
         }
         if (m1 >= 0) {
            this->d_E_[m1] += val_4*(s_m3*s_m2 + s_m2*s_p1 + s_p1*s_p2) + val_3*(s_m2 + s_p1) + val_2;
         }
         if (p1 < this->system_size_) {
            this->d_E_[p1] += val_4*(s_m2*s_m1 + s_m1*s_p2 + s_p2*s_p3) + val_3*(s_m1 + s_p2) + val_2;
         }
         if (p2 < this->system_size_) {
            this->d_E_[p2] += val_4*(s_m1*s_p1 + s_p1*s_p3) + val_3*s_p1;
         }
         if (p3 < this->system_size_) {
            this->d_E_[p3] += val_4*s_p1*s_p2;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip5Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = (interaction_.count(2) == 1 ? interaction_.at(2) : 0)*diff;
      const double val_3 = (interaction_.count(3) == 1 ? interaction_.at(3) : 0)*diff;
      const double val_4 = (interaction_.count(4) == 1 ? interaction_.at(4) : 0)*diff;
      const double val_5 = interaction_.at(5)*diff;
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         const std::int32_t m4 = (index - 4 + this->system_size_)%this->system_size_;
         const std::int32_t m3 = (index - 3 + this->system_size_)%this->system_size_;
         const std::int32_t m2 = (index - 2 + this->system_size_)%this->system_size_;
         const std::int32_t m1 = (index - 1 + this->system_size_)%this->system_size_;
         const std::int32_t p1 = (index + 1)%this->system_size_;
         const std::int32_t p2 = (index + 2)%this->system_size_;
         const std::int32_t p3 = (index + 3)%this->system_size_;
         const std::int32_t p4 = (index + 4)%this->system_size_;
         const double s_m4 = this->sample_[m4].GetValue();
         const double s_m3 = this->sample_[m3].GetValue();
         const double s_m2 = this->sample_[m2].GetValue();
         const double s_m1 = this->sample_[m1].GetValue();
         const double s_p1 = this->sample_[p1].GetValue();
         const double s_p2 = this->sample_[p2].GetValue();
         const double s_p3 = this->sample_[p3].GetValue();
         const double s_p4 = this->sample_[p4].GetValue();
         this->d_E_[m4] += val_5*s_m3*s_m2*s_m1;
         this->d_E_[m3] += val_5*(s_m4*s_m2*s_m1 + s_m2*s_m1*s_p1) + val_4*s_m2*s_m1;
         this->d_E_[m2] += val_5*(s_m4*s_m3*s_m1 + s_m3*s_m1*s_p1 + s_m1*s_p1*s_p2) + val_4*(s_m3*s_m1 + s_m1*s_p1) + val_3*s_m1;
         this->d_E_[m1] += val_5*(s_m4*s_m3*s_m2 + s_m3*s_m2*s_p1 + s_m2*s_p1*s_p2 + s_p1*s_p2*s_p3) + val_4*(s_m3*s_m2 + s_m2*s_p1 + s_p1*s_p2) + val_3*(s_m2 + s_p1) + val_2;
         this->d_E_[p1] += val_5*(s_m3*s_m2*s_m1 + s_m2*s_m1*s_p2 + s_m1*s_p2*s_p3 + s_p2*s_p3*s_p4) + val_4*(s_m2*s_m1 + s_m1*s_p2 + s_p2*s_p3) + val_3*(s_m1 + s_p2) + val_2;
         this->d_E_[p2] += val_5*(s_m2*s_m1*s_p1 + s_m1*s_p1*s_p3 + s_p1*s_p3*s_p4) + val_4*(s_m1*s_p1 + s_p1*s_p3) + val_3*s_p1;
         this->d_E_[p3] += val_5*(s_m1*s_p1*s_p2 + s_p1*s_p2*s_p4) + val_4*s_p1*s_p2;
         this->d_E_[p4] += val_5*s_p1*s_p2*s_p3;
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         const std::int32_t m4 = index - 4;
         const std::int32_t m3 = index - 3;
         const std::int32_t m2 = index - 2;
         const std::int32_t m1 = index - 1;
         const std::int32_t p1 = index + 1;
         const std::int32_t p2 = index + 2;
         const std::int32_t p3 = index + 3;
         const std::int32_t p4 = index + 4;
         const double s_m4 = m4 >= 0 ? this->sample_[m4].GetValue() : 0;
         const double s_m3 = m3 >= 0 ? this->sample_[m3].GetValue() : 0;
         const double s_m2 = m2 >= 0 ? this->sample_[m2].GetValue() : 0;
         const double s_m1 = m1 >= 0 ? this->sample_[m1].GetValue() : 0;
         const double s_p1 = p1 < this->system_size_ ? this->sample_[p1].GetValue() : 0;
         const double s_p2 = p2 < this->system_size_ ? this->sample_[p2].GetValue() : 0;
         const double s_p3 = p3 < this->system_size_ ? this->sample_[p3].GetValue() : 0;
         const double s_p4 = p4 < this->system_size_ ? this->sample_[p4].GetValue() : 0;
         if (m4 >= 0) {
            this->d_E_[m4] += val_5*s_m3*s_m2*s_m1;
         }
         if (m3 >= 0) {
            this->d_E_[m3] += val_5*(s_m4*s_m2*s_m1 + s_m2*s_m1*s_p1) + val_4*s_m2*s_m1;
         }
         if (m2 >= 0) {
            this->d_E_[m2] += val_5*(s_m4*s_m3*s_m1 + s_m3*s_m1*s_p1 + s_m1*s_p1*s_p2) + val_4*(s_m3*s_m1 + s_m1*s_p1) + val_3*s_m1;
         }
         if (m1 >= 0) {
            this->d_E_[m1] += val_5*(s_m4*s_m3*s_m2 + s_m3*s_m2*s_p1 + s_m2*s_p1*s_p2 + s_p1*s_p2*s_p3) + val_4*(s_m3*s_m2 + s_m2*s_p1 + s_p1*s_p2) + val_3*(s_m2 + s_p1) + val_2;
         }
         if (p1 < this->system_size_) {
            this->d_E_[p1] += val_5*(s_m3*s_m2*s_m1 + s_m2*s_m1*s_p2 + s_m1*s_p2*s_p3 + s_p2*s_p3*s_p4) + val_4*(s_m2*s_m1 + s_m1*s_p2 + s_p2*s_p3) + val_3*(s_m1 + s_p2) + val_2;
         }
         if (p2 < this->system_size_) {
            this->d_E_[p2] += val_5*(s_m2*s_m1*s_p1 + s_m1*s_p1*s_p3 + s_p1*s_p3*s_p4) + val_4*(s_m1*s_p1 + s_p1*s_p3) + val_3*s_p1;
         }
         if (p3 < this->system_size_) {
            this->d_E_[p3] += val_5*(s_m1*s_p1*s_p2 + s_p1*s_p2*s_p4) + val_4*s_p1*s_p2;
         }
         if (p4 < this->system_size_) {
            this->d_E_[p4] += val_5*s_p1*s_p2*s_p3;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void FlipAny(const std::int32_t index, const std::int32_t update_state) {
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         for (const auto &it: interaction_) {
            const double val = it.second*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
            for (std::int32_t i = 0; i < it.first; ++i) {
               double spin_prod = 1;
               std::int32_t num_zero = 0;
               for (std::int32_t j = 0; j < it.first; ++j) {
                  const std::int32_t p_ind = (index - it.first + 1 + i + j + this->system_size_)%this->system_size_;
                  if (p_ind != index) {
                     if (this->sample_[p_ind].GetValue() != 0.0) {
                        spin_prod *= this->sample_[p_ind].GetValue();
                     }
                     else {
                        num_zero++;
                     }
                  }
               }
               if (num_zero == 0) {
                  for (std::int32_t j = 0; j < it.first; ++j) {
                     const std::int32_t p_ind = (index - it.first + 1 + i + j + this->system_size_)%this->system_size_;
                     if (p_ind != index) {
                        this->d_E_[p_ind] += val*spin_prod/this->sample_[p_ind].GetValue();
                     }
                  }
               }
               else if (num_zero == 1) {
                  for (std::int32_t j = 0; j < it.first; ++j) {
                     const std::int32_t p_ind = (index - it.first + 1 + i + j + this->system_size_)%this->system_size_;
                     if (p_ind != index) {
                        if (this->sample_[p_ind].GetValue() == 0.0) {
                           this->d_E_[p_ind] += val*spin_prod;
                        }
                     }
                  }
               }
            }
         }
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            const double val = it.second*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
            for (std::int32_t i = std::max(index - it.first + 1, 0); i <= index; ++i) {
               if (i > this->system_size_ - it.first) {
                  break;
               }
               double spin_prod = 1;
               std::int32_t num_zero = 0;
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != index) {
                     if (this->sample_[j].GetValue() != 0.0) {
                        spin_prod *= this->sample_[j].GetValue();
                     }
                     else {
                        num_zero++;
                     }
                  }
               }
               if (num_zero == 0) {
                  for (std::int32_t j = i; j < i + it.first; ++j) {
                     if (j != index) {
                        this->d_E_[j] += val*spin_prod/this->sample_[j].GetValue();
                     }
                  }
               }
               else if (num_zero == 1) {
                  for (std::int32_t j = i; j < i + it.first; ++j) {
                     if (j != index) {
                        if (this->sample_[j].GetValue() == 0.0) {
                           this->d_E_[j] += val*spin_prod;
                        }
                     }
                  }
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
};



} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
