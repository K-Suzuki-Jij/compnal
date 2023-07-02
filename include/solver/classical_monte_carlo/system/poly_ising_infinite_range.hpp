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
//  poly_ising_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2023/06/29.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_INFINITE_RANGE_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_INFINITE_RANGE_HPP_

#include "template_system.hpp"
#include "../../../utility/combination.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/infinite_range.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief System class for the Ising model on a chain.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::PolynomialIsing<lattice::InfiniteRange>, RandType>: public BaseIsingSystem<model::classical::PolynomialIsing<lattice::InfiniteRange>, RandType> {
   
   //! @brief Model type.
   using ModelType = model::classical::PolynomialIsing<lattice::InfiniteRange>;
   
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   interaction_(model.GetInteraction()),
   degree_(model.GetDegree()) {
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
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
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
      else if (degree_ == 6) {
         Flip6Body(index, update_state);
      }
      else if (degree_ == 7) {
         Flip7Body(index, update_state);
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
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const std::vector<model::utility::Spin> &sample) const {
      std::vector<double> d_E_(this->system_size_);
      
      for (std::int32_t index = 0; index < this->system_size_; ++index) {
         double val = 0.0;
         for (const auto &it: interaction_) {
            if (it.first < 2) {
               if (it.first == 1) {
                  val += it.second;
               }
               continue;
            }
            std::vector<std::int32_t> indices(it.first - 1);
            std::int32_t start_index = 0;
            std::int32_t size = 0;
            
            while (true) {
               for (std::int32_t i = start_index; i < this->system_size_ - 1; ++i) {
                  indices[size++] = i;
                  if (size == it.first - 1) {
                     double spin_prod = 1;
                     for (std::int32_t j = 0; j < it.first - 1; ++j) {
                        if (indices[j] >= index) {
                           spin_prod *= sample[indices[j] + 1].GetValue();
                        }
                        else {
                           spin_prod *= sample[indices[j]].GetValue();
                        }
                     }
                     val += it.second*spin_prod;
                     break;
                  }
               }
               --size;
               if (size < 0) {
                  break;
               }
               start_index = indices[size] + 1;
            }
         }
         d_E_[index] += val;
      }      
      return d_E_;
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip2Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = interaction_.at(2)*diff;
      for (std::int32_t i2 = 0; i2 < this->system_size_; ++i2) {
         if (i2 == index) {continue;}
         this->d_E_[i2] += val_2;
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
      double all_sum = 0;
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         all_sum += this->sample_[i].GetValue();
      }
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         this->d_E_[i] += val_3*(all_sum - this->sample_[i].GetValue()) + val_2;;
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
      double all_sum = 0;
      double square_sum = 0;
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         all_sum += this->sample_[i].GetValue();
         square_sum += this->sample_[i].GetValue()*this->sample_[i].GetValue();
      }
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double a = (all_sum - this->sample_[i].GetValue());
         const double b = (square_sum - this->sample_[i].GetValue()*this->sample_[i].GetValue());
         this->d_E_[i] += val_4*(a*a - b)*0.5 + val_3*a + val_2;
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
      double all_sum = 0;
      double square_sum = 0;
      double cubic_sum = 0;
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double s = this->sample_[i].GetValue();
         all_sum += s;
         square_sum += s*s;
         cubic_sum += s*s*s;
      }
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double s = this->sample_[i].GetValue();
         const double a1 = all_sum - s;
         const double a2 = square_sum - s*s;
         const double a3 = cubic_sum - s*s*s;
         this->d_E_[i] += val_5*(a1*a1*a1 - 3*a2*a1 + 2*a3)/6 + val_4*(a1*a1 - a2)/2 + val_3*a1 + val_2;
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip6Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = (interaction_.count(2) == 1 ? interaction_.at(2) : 0)*diff;
      const double val_3 = (interaction_.count(3) == 1 ? interaction_.at(3) : 0)*diff;
      const double val_4 = (interaction_.count(4) == 1 ? interaction_.at(4) : 0)*diff;
      const double val_5 = (interaction_.count(5) == 1 ? interaction_.at(5) : 0)*diff;
      const double val_6 = interaction_.at(6)*diff;
      double m1 = 0;
      double m2 = 0;
      double m3 = 0;
      double m4 = 0;
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double s = this->sample_[i].GetValue();
         m1 += s;
         m2 += s*s;
         m3 += s*s*s;
         m4 += s*s*s*s;
      }
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double s = this->sample_[i].GetValue();
         const double a1 = m1 - s;
         const double a2 = m2 - s*s;
         const double a3 = m3 - s*s*s;
         const double a4 = m4 - s*s*s*s;
         const double diff1 = val_3*a1;
         const double diff2 = val_4*(a1*a1 - a2)/2.0;
         const double diff3 = val_5*(a1*a1*a1 - 3*a2*a1 + 2*a3)/6.0;
         const double diff4 = val_6*(a1*a1*a1*a1 - 6*a4 + 3*a2*a2 + 8*a3*a1 - 6*a2*a1*a1)/24.0;
         this->d_E_[i] += diff4 + diff3 + diff2 + diff1 + val_2;
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip7Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = (interaction_.count(2) == 1 ? interaction_.at(2) : 0)*diff;
      const double val_3 = (interaction_.count(3) == 1 ? interaction_.at(3) : 0)*diff;
      const double val_4 = (interaction_.count(4) == 1 ? interaction_.at(4) : 0)*diff;
      const double val_5 = (interaction_.count(5) == 1 ? interaction_.at(5) : 0)*diff;
      const double val_6 = (interaction_.count(6) == 1 ? interaction_.at(6) : 0)*diff;
      const double val_7 = interaction_.at(7)*diff;
      double m1 = 0;
      double m2 = 0;
      double m3 = 0;
      double m4 = 0;
      double m5 = 0;
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double s = this->sample_[i].GetValue();
         m1 += s;
         m2 += s*s;
         m3 += s*s*s;
         m4 += s*s*s*s;
         m5 += s*s*s*s*s;
      }
      for (std::int32_t i = 0; i < this->system_size_; ++i) {
         if (i == index) {continue;}
         const double s = this->sample_[i].GetValue();
         const double a1 = m1 - s;
         const double a2 = m2 - s*s;
         const double a3 = m3 - s*s*s;
         const double a4 = m4 - s*s*s*s;
         const double a5 = m5 - s*s*s*s*s;
         const double diff1 = val_3*a1;
         const double diff2 = val_4*(a1*a1 - a2)/2.0;
         const double diff3 = val_5*(a1*a1*a1 - 3*a2*a1 + 2*a3)/6.0;
         const double diff4 = val_6*(a1*a1*a1*a1 - 6*a4 + 3*a2*a2 + 8*a3*a1 - 6*a2*a1*a1)/24.0;
         const double diff5 = val_7*(a1*a1*a1*a1*a1 + 24*a5 - 30*a1*a4 + 20*a1*a1*a3 + 15*a1*a2*a2 - 10*a1*a1*a1*a2 - 20*a2*a3)/120.0;
         this->d_E_[i] += diff5 + diff4 + diff3 + diff2 + diff1 + val_2;
      }
      this->sample_[index].SetState(update_state);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void FlipAny(const std::int32_t index, const std::int32_t update_state) {
      for (const auto &it: interaction_) {
         if (it.first <= 1) {
            continue;
         }
         const double val = it.second*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
         std::vector<std::int32_t> indices(it.first);
         std::int32_t start_index = 0;
         std::int32_t size = 0;
         
         while (true) {
            for (std::int32_t i = start_index; i < this->system_size_ - 1; ++i) {
               indices[size++] = i;
               if (size == it.first - 1) {
                  double spin_prod = 1;
                  for (std::int32_t j = 0; j < it.first - 1; ++j) {
                     if (indices[j] >= index) {
                        spin_prod *= this->sample_[indices[j] + 1].GetValue();
                     }
                     else {
                        spin_prod *= this->sample_[indices[j]].GetValue();
                     }
                  }

                  for (std::int32_t j = 0; j < it.first - 1; ++j) {
                     if (indices[j] >= index) {
                        this->d_E_[indices[j] + 1] += val*spin_prod/this->sample_[indices[j] + 1].GetValue();
                     }
                     else {
                        this->d_E_[indices[j]] += val*spin_prod/this->sample_[indices[j]].GetValue();
                     }
                  }
                  break;
               }
            }
            --size;
            if (size < 0) {
               break;
            }
            start_index = indices[size] + 1;
         }
      }
      this->sample_[index].SetState(update_state);
   }
   
};

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_INFINITE_RANGE_HPP_ */
