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
//  poly_ising_cubic.hpp
//  compnal
//
//  Created by kohei on 2023/06/29.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_CUBIC_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_CUBIC_HPP_

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief System class for the Ising model on a chain.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::PolynomialIsing<lattice::Cubic>, RandType>: public BaseIsingSystem<model::classical::PolynomialIsing<lattice::Cubic>, RandType> {
   
   //! @brief Model type.
   using ModelType = model::classical::PolynomialIsing<lattice::Cubic>;

   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   interaction_(model.GetInteraction()),
   degree_(model.GetDegree()),
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
      else {
         FlipAny(index, update_state);
      }
   }
   
private:
   //! @brief The polynomial interaction.
   std::unordered_map<std::int32_t, double> interaction_;
   
   //! @brief The degree of the interactions.
   std::int32_t degree_ = 0;
   
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
                  double val = 0;
                  for (const auto &it: interaction_) {
                     double spin_prod_x = 1;
                     double spin_prod_y = 1;
                     double spin_prod_z = 1;
                     for (std::int32_t diff = 1; diff < it.first; ++diff) {
                        const std::int32_t i_x = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - diff + x_size_)%x_size_;
                        const std::int32_t i_y = coo_z*x_size_*y_size_ + ((coo_y - diff + y_size_)%y_size_)*x_size_ + coo_x;
                        const std::int32_t i_z = ((coo_z - diff + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                        spin_prod_x *= this->sample_[i_x].GetValue();
                        spin_prod_y *= this->sample_[i_y].GetValue();
                        spin_prod_z *= this->sample_[i_z].GetValue();
                     }
                     if (it.first == 1) {
                        val += it.second;
                     }
                     else {
                        for (std::int32_t diff = it.first - 1; diff >= 0; --diff) {
                           val += it.second*(spin_prod_x + spin_prod_y + spin_prod_z);
                           const std::int32_t i_x1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - diff + x_size_)%x_size_;
                           const std::int32_t i_x2 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - diff + it.first + x_size_)%x_size_;
                           const std::int32_t i_y1 = coo_z*x_size_*y_size_ + ((coo_y - diff + y_size_)%y_size_)*x_size_ + coo_x;
                           const std::int32_t i_y2 = coo_z*x_size_*y_size_ + ((coo_y - diff + it.first + y_size_)%y_size_)*x_size_ + coo_x;
                           const std::int32_t i_z1 = ((coo_z - diff + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                           const std::int32_t i_z2 = ((coo_z - diff + it.first + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                           spin_prod_x = spin_prod_x*this->sample_[i_x2].GetValue()/this->sample_[i_x1].GetValue();
                           spin_prod_y = spin_prod_y*this->sample_[i_y2].GetValue()/this->sample_[i_y1].GetValue();
                           spin_prod_z = spin_prod_z*this->sample_[i_z2].GetValue()/this->sample_[i_z1].GetValue();
                        }
                     }
                  }
                  d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val;
               }
            }
         }
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size_; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size_; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size_; ++coo_z) {
                  double val = 0;
                  for (const auto &it: interaction_) {
                     double spin_prod_x = 1;
                     double spin_prod_y = 1;
                     double spin_prod_z = 1;
                     for (std::int32_t diff = 1; diff < it.first; ++diff) {
                        const std::int32_t i_x = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - diff + x_size_)%x_size_;
                        const std::int32_t i_y = coo_z*x_size_*y_size_ + ((coo_y - diff + y_size_)%y_size_)*x_size_ + coo_x;
                        const std::int32_t i_z = ((coo_z - diff + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                        spin_prod_x *= this->sample_[i_x].GetValue();
                        spin_prod_y *= this->sample_[i_y].GetValue();
                        spin_prod_z *= this->sample_[i_z].GetValue();
                     }
                     if (it.first == 1) {
                        val += it.second;
                     }
                     else {
                        for (std::int32_t diff = it.first - 1; diff >= 0; --diff) {
                           if (coo_x - diff >= 0 && coo_x - diff + it.first - 1 < x_size_) {
                              val += it.second*spin_prod_x;
                           }
                           if (coo_y - diff >= 0 && coo_y - diff + it.first - 1 < y_size_) {
                              val += it.second*spin_prod_y;
                           }
                           if (coo_z - diff >= 0 && coo_z - diff + it.first - 1 < z_size_) {
                              val += it.second*spin_prod_z;
                           }
                           const std::int32_t i_x1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - diff + x_size_)%x_size_;
                           const std::int32_t i_x2 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - diff + it.first + x_size_)%x_size_;
                           const std::int32_t i_y1 = coo_z*x_size_*y_size_ + ((coo_y - diff + y_size_)%y_size_)*x_size_ + coo_x;
                           const std::int32_t i_y2 = coo_z*x_size_*y_size_ + ((coo_y - diff + it.first + y_size_)%y_size_)*x_size_ + coo_x;
                           const std::int32_t i_z1 = ((coo_z - diff + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                           const std::int32_t i_z2 = ((coo_z - diff + it.first + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                           spin_prod_x = spin_prod_x*this->sample_[i_x2].GetValue()/this->sample_[i_x1].GetValue();
                           spin_prod_y = spin_prod_y*this->sample_[i_y2].GetValue()/this->sample_[i_y1].GetValue();
                           spin_prod_z = spin_prod_z*this->sample_[i_z2].GetValue()/this->sample_[i_z1].GetValue();
                        }
                     }
                  }
                  d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val;
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      return d_E_;
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip2Body(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue();
      const double val_2 = interaction_.at(2)*diff;
      const std::int32_t coo_x = index%x_size_;
      const std::int32_t coo_y = (index%(x_size_*y_size_))/x_size_;
      const std::int32_t coo_z = index/(x_size_*y_size_);
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - 1 + x_size_)%x_size_] += val_2;
         this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x + 1)%x_size_] += val_2;
         this->d_E_[coo_z*x_size_*y_size_ + ((coo_y - 1 + y_size_)%y_size_)*x_size_ + coo_x] += val_2;
         this->d_E_[coo_z*x_size_*y_size_ + ((coo_y + 1)%y_size_)*x_size_ + coo_x] += val_2;
         this->d_E_[((coo_z - 1 + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val_2;
         this->d_E_[((coo_z + 1)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val_2;
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         if (coo_x - 1 >= 0) {
            this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x - 1] += val_2;
         }
         if (coo_y - 1 >= 0) {
            this->d_E_[coo_z*x_size_*y_size_ + (coo_y - 1)*x_size_ + coo_x] += val_2;
         }
         if (coo_z - 1 >= 0) {
            this->d_E_[(coo_z - 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val_2;
         }
         if (coo_x + 1 < x_size_) {
            this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x + 1] += val_2;
         }
         if (coo_y + 1 < y_size_) {
            this->d_E_[coo_z*x_size_*y_size_ + (coo_y + 1)*x_size_ + coo_x] += val_2;
         }
         if (coo_z + 1 < z_size_) {
            this->d_E_[(coo_z + 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val_2;
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
      const std::int32_t coo_x = index%x_size_;
      const std::int32_t coo_y = (index%(x_size_*y_size_))/x_size_;
      const std::int32_t coo_z = index/(x_size_*y_size_);
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         const std::int32_t x_m2 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - 2 + x_size_)%x_size_;
         const std::int32_t x_m1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - 1 + x_size_)%x_size_;
         const std::int32_t x_p1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x + 1)%x_size_;
         const std::int32_t x_p2 = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x + 2)%x_size_;
         const std::int32_t y_m2 = coo_z*x_size_*y_size_ + ((coo_y - 2 + y_size_)%y_size_)*x_size_ + coo_x;
         const std::int32_t y_m1 = coo_z*x_size_*y_size_ + ((coo_y - 1 + y_size_)%y_size_)*x_size_ + coo_x;
         const std::int32_t y_p1 = coo_z*x_size_*y_size_ + ((coo_y + 1)%y_size_)*x_size_ + coo_x;
         const std::int32_t y_p2 = coo_z*x_size_*y_size_ + ((coo_y + 2)%y_size_)*x_size_ + coo_x;
         const std::int32_t z_m2 = ((coo_z - 2 + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const std::int32_t z_m1 = ((coo_z - 1 + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const std::int32_t z_p1 = ((coo_z + 1)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const std::int32_t z_p2 = ((coo_z + 2)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         this->d_E_[x_m2] += val_3*this->sample_[x_m1].GetValue();
         this->d_E_[x_m1] += val_3*(this->sample_[x_m2].GetValue() + this->sample_[x_p1].GetValue()) + val_2;
         this->d_E_[x_p1] += val_3*(this->sample_[x_m1].GetValue() + this->sample_[x_p2].GetValue()) + val_2;
         this->d_E_[x_p2] += val_3*this->sample_[x_p1].GetValue();
         this->d_E_[y_m2] += val_3*this->sample_[y_m1].GetValue();
         this->d_E_[y_m1] += val_3*(this->sample_[y_m2].GetValue() + this->sample_[y_p1].GetValue()) + val_2;
         this->d_E_[y_p1] += val_3*(this->sample_[y_m1].GetValue() + this->sample_[y_p2].GetValue()) + val_2;
         this->d_E_[y_p2] += val_3*this->sample_[y_p1].GetValue();
         this->d_E_[z_m2] += val_3*this->sample_[z_m1].GetValue();
         this->d_E_[z_m1] += val_3*(this->sample_[z_m2].GetValue() + this->sample_[z_p1].GetValue()) + val_2;
         this->d_E_[z_p1] += val_3*(this->sample_[z_m1].GetValue() + this->sample_[z_p2].GetValue()) + val_2;
         this->d_E_[z_p2] += val_3*this->sample_[z_p1].GetValue();
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         const std::int32_t x_m2 = coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x - 2;
         const std::int32_t x_m1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x - 1;
         const std::int32_t x_p1 = coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x + 1;
         const std::int32_t x_p2 = coo_z*x_size_*y_size_ + coo_y*x_size_ + coo_x + 2;
         const std::int32_t y_m2 = coo_z*x_size_*y_size_ + (coo_y - 2)*x_size_ + coo_x;
         const std::int32_t y_m1 = coo_z*x_size_*y_size_ + (coo_y - 1)*x_size_ + coo_x;
         const std::int32_t y_p1 = coo_z*x_size_*y_size_ + (coo_y + 1)*x_size_ + coo_x;
         const std::int32_t y_p2 = coo_z*x_size_*y_size_ + (coo_y + 2)*x_size_ + coo_x;
         const std::int32_t z_m2 = (coo_z - 2)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const std::int32_t z_m1 = (coo_z - 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const std::int32_t z_p1 = (coo_z + 1)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const std::int32_t z_p2 = (coo_z + 2)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
         const double x_s_m2 = coo_x - 2 >= 0 ? this->sample_[x_m2].GetValue() : 0;
         const double x_s_m1 = coo_x - 1 >= 0 ? this->sample_[x_m1].GetValue() : 0;
         const double x_s_p1 = coo_x + 1 < x_size_ ? this->sample_[x_p1].GetValue() : 0;
         const double x_s_p2 = coo_x + 2 < x_size_ ? this->sample_[x_p2].GetValue() : 0;
         const double y_s_m2 = coo_y - 2 >= 0 ? this->sample_[y_m2].GetValue() : 0;
         const double y_s_m1 = coo_y - 1 >= 0 ? this->sample_[y_m1].GetValue() : 0;
         const double y_s_p1 = coo_y + 1 < y_size_ ? this->sample_[y_p1].GetValue() : 0;
         const double y_s_p2 = coo_y + 2 < y_size_ ? this->sample_[y_p2].GetValue() : 0;
         const double z_s_m2 = coo_z - 2 >= 0 ? this->sample_[z_m2].GetValue() : 0;
         const double z_s_m1 = coo_z - 1 >= 0 ? this->sample_[z_m1].GetValue() : 0;
         const double z_s_p1 = coo_z + 1 < z_size_ ? this->sample_[z_p1].GetValue() : 0;
         const double z_s_p2 = coo_z + 2 < z_size_ ? this->sample_[z_p2].GetValue() : 0;
         if (coo_x - 2 >= 0) {
            this->d_E_[x_m2] += val_3*x_s_m1;
         }
         if (coo_x - 1 >= 0) {
            this->d_E_[x_m1] += val_3*(x_s_m2 + x_s_p1) + val_2;
         }
         if (coo_x + 1 < x_size_) {
            this->d_E_[x_p1] += val_3*(x_s_m1 + x_s_p2) + val_2;
         }
         if (coo_x + 2 < x_size_) {
            this->d_E_[x_p2] += val_3*x_s_p1;
         }
         if (coo_y - 2 >= 0) {
            this->d_E_[y_m2] += val_3*y_s_m1;
         }
         if (coo_y - 1 >= 0) {
            this->d_E_[y_m1] += val_3*(y_s_m2 + y_s_p1) + val_2;
         }
         if (coo_y + 1 < y_size_) {
            this->d_E_[y_p1] += val_3*(y_s_m1 + y_s_p2) + val_2;
         }
         if (coo_y + 2 < y_size_) {
            this->d_E_[y_p2] += val_3*y_s_p1;
         }
         if (coo_z - 2 >= 0) {
            this->d_E_[z_m2] += val_3*z_s_m1;
         }
         if (coo_z - 1 >= 0) {
            this->d_E_[z_m1] += val_3*(z_s_m2 + z_s_p1) + val_2;
         }
         if (coo_z + 1 < z_size_) {
            this->d_E_[z_p1] += val_3*(z_s_m1 + z_s_p2) + val_2;
         }
         if (coo_z + 2 < z_size_) {
            this->d_E_[z_p2] += val_3*z_s_p1;
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
      const std::int32_t coo_x = index%x_size_;
      const std::int32_t coo_y = (index%(x_size_*y_size_))/x_size_;
      const std::int32_t coo_z = index/(x_size_*y_size_);
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         for (const auto &it: interaction_) {
            const double val = it.second*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
            for (std::int32_t i = 0; i < it.first; ++i) {
               double spin_prod_x = 1;
               double spin_prod_y = 1;
               double spin_prod_z = 1;
               for (std::int32_t j = 0; j < it.first; ++j) {
                  const std::int32_t p_ind_x = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - it.first + 1 + i + j + x_size_)%x_size_;
                  const std::int32_t p_ind_y = coo_z*x_size_*y_size_ + ((coo_y - it.first + 1 + i + j + y_size_)%y_size_)*x_size_ + coo_x;
                  const std::int32_t p_ind_z = ((coo_z - it.first + 1 + i + j + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                  if (p_ind_x != index) {
                     spin_prod_x *= this->sample_[p_ind_x].GetValue();
                  }
                  if (p_ind_y != index) {
                     spin_prod_y *= this->sample_[p_ind_y].GetValue();
                  }
                  if (p_ind_z != index) {
                     spin_prod_z *= this->sample_[p_ind_z].GetValue();
                  }
               }
               for (std::int32_t j = 0; j < it.first; ++j) {
                  const std::int32_t p_ind_x = coo_z*x_size_*y_size_ + coo_y*x_size_ + (coo_x - it.first + 1 + i + j + x_size_)%x_size_;
                  const std::int32_t p_ind_y = coo_z*x_size_*y_size_ + ((coo_y - it.first + 1 + i + j + y_size_)%y_size_)*x_size_ + coo_x;
                  const std::int32_t p_ind_z = ((coo_z - it.first + 1 + i + j + z_size_)%z_size_)*x_size_*y_size_ + coo_y*x_size_ + coo_x;
                  if (p_ind_x != index) {
                     this->d_E_[p_ind_x] += val*spin_prod_x/this->sample_[p_ind_x].GetValue();
                  }
                  if (p_ind_y != index) {
                     this->d_E_[p_ind_y] += val*spin_prod_y/this->sample_[p_ind_y].GetValue();
                  }
                  if (p_ind_z != index) {
                     this->d_E_[p_ind_z] += val*spin_prod_z/this->sample_[p_ind_z].GetValue();
                  }
               }
            }
         }
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            const double val = it.second*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
            
            // x-direction
            for (std::int32_t i = std::max(coo_x - it.first + 1, 0); i <= coo_x; ++i) {
               if (i > x_size_ - it.first) {
                  break;
               }
               double spin_prod = 1;
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != coo_x) {
                     spin_prod *= this->sample_[coo_z*x_size_*y_size_ + coo_y*x_size_ + j].GetValue();
                  }
               }
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != coo_x) {
                     this->d_E_[coo_z*x_size_*y_size_ + coo_y*x_size_ + j] += val*spin_prod/this->sample_[coo_z*x_size_*y_size_ + coo_y*x_size_ + j].GetValue();
                  }
               }
            }
            
            // y-direction
            for (std::int32_t i = std::max(coo_y - it.first + 1, 0); i <= coo_y; ++i) {
               if (i > y_size_ - it.first) {
                  break;
               }
               double spin_prod = 1;
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != coo_y) {
                     spin_prod *= this->sample_[coo_z*x_size_*y_size_ + j*x_size_ + coo_x].GetValue();
                  }
               }
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != coo_y) {
                     this->d_E_[coo_z*x_size_*y_size_ + j*x_size_ + coo_x] += val*spin_prod/this->sample_[coo_z*x_size_*y_size_ + j*x_size_ + coo_x].GetValue();
                  }
               }
            }
            
            // z-direction
            for (std::int32_t i = std::max(coo_z - it.first + 1, 0); i <= coo_z; ++i) {
               if (i > z_size_ - it.first) {
                  break;
               }
               double spin_prod = 1;
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != coo_z) {
                     spin_prod *= this->sample_[j*x_size_*y_size_ + coo_y*x_size_ + coo_x].GetValue();
                  }
               }
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  if (j != coo_z) {
                     this->d_E_[j*x_size_*y_size_ + coo_y*x_size_ + coo_x] += val*spin_prod/this->sample_[j*x_size_*y_size_ + coo_y*x_size_ + coo_x].GetValue();
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


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_CUBIC_HPP_ */
