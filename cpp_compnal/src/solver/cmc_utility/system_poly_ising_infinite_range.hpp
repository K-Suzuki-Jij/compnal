//
//  Copyright 2022 Kohei Suzuki
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
//  system_poly_ising_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_POLY_ISING_INFINITE_RANGE_HPP_
#define COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_POLY_ISING_INFINITE_RANGE_HPP_

#include "../../lattice/all.hpp"
#include "../../model/all.hpp"
#include "base_system.hpp"

namespace compnal {
namespace solver {
namespace cmc_utility {

template<typename RealType>
class CMCSystem<model::PolynomialIsing<lattice::InfiniteRange, RealType>>: public CMCBaseIsingSystem {
   
   using ModelType = model::PolynomialIsing<lattice::InfiniteRange, RealType>;
   
public:
   using ValueType = typename ModelType::ValueType;
   
   CMCSystem(const ModelType &model):
   system_size_(model.GetSystemSize()),
   bc_(model.GetBoundaryCondition()),
   interaction_(model.GetInteraction()),
   degree_(model.GetDegree()){}
   
   void InitializeSSF(const uint64_t seed) {
      sample_ = this->GenerateRandomSpin(seed, system_size_);
      energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   void Flip(const std::int32_t index) {
      const typename ModelType::OPType target_spin = sample_[index];
      if (degree_ == 2) {
         const typename ModelType::ValueType target_ineraction_deg2 = interaction_[2];
         for (std::int32_t i2 = 0; i2 < system_size_; ++i2) {
            if (i2 == index) {continue;}
            energy_difference_[i2] += 4*target_ineraction_deg2*target_spin*sample_[i2];
         }
      }
      else if (degree_ == 3) {
         const typename ModelType::ValueType target_ineraction_deg2 = interaction_[2];
         const typename ModelType::ValueType target_ineraction_deg3 = interaction_[3];
         const typename ModelType::OPType spin_prod1 = target_spin;
         for (std::int32_t i2 = 0; i2 < system_size_; ++i2) {
            if (i2 == index) {continue;}
            const typename ModelType::OPType spin_prod2 = spin_prod1*sample_[i2];
            const typename ModelType::ValueType diff_val_i2 = 4*target_ineraction_deg2*spin_prod2;
            typename ModelType::ValueType sum_val_i2 = 0;
            for (std::int32_t i3 = i2 + 1; i3 < index; ++i3) {
               const typename ModelType::ValueType diff_val_i3 = 4*target_ineraction_deg3*spin_prod2*sample_[i3];
               energy_difference_[i3] += diff_val_i3;
               sum_val_i2 += diff_val_i3;
            }
            for (std::int32_t i3 = std::max(index + 1, i2 + 1); i3 < system_size_; ++i3) {
               const typename ModelType::ValueType diff_val_i3 = 4*target_ineraction_deg3*spin_prod2*sample_[i3];
               energy_difference_[i3] += diff_val_i3;
               sum_val_i2 += diff_val_i3;
            }
            energy_difference_[i2] += sum_val_i2 + diff_val_i2;
         }
      }
      else if (degree_ == 4) {
         const typename ModelType::ValueType target_ineraction_deg2 = interaction_[2];
         const typename ModelType::ValueType target_ineraction_deg3 = interaction_[3];
         const typename ModelType::ValueType target_ineraction_deg4 = interaction_[4];
         const typename ModelType::OPType spin_prod1 = target_spin;
         for (std::int32_t i2 = 0; i2 < system_size_; ++i2) {
            if (i2 == index) {continue;}
            const typename ModelType::OPType spin_prod2 = spin_prod1*sample_[i2];
            const typename ModelType::ValueType diff_val_i2 = 4*target_ineraction_deg2*spin_prod2;
            typename ModelType::ValueType sum_val_i2 = 0;
            for (std::int32_t i3 = i2 + 1; i3 < system_size_; ++i3) {
               if (i3 == index) {continue;}
               const typename ModelType::OPType spin_prod3 = spin_prod2*sample_[i3];
               const typename ModelType::ValueType diff_val_i3 = 4*target_ineraction_deg3*spin_prod3;
               typename ModelType::ValueType sum_val_i3 = 0;
               for (std::int32_t i4 = i3 + 1; i4 < index; ++i4) {
                  const typename ModelType::ValueType diff_val_i4 = 4*target_ineraction_deg4*spin_prod3*sample_[i4];
                  energy_difference_[i4] += diff_val_i4;
                  sum_val_i3 += diff_val_i4;
               }
               for (std::int32_t i4 = std::max(index + 1, i3 + 1); i4 < system_size_; ++i4) {
                  const typename ModelType::ValueType diff_val_i4 = 4*target_ineraction_deg4*spin_prod3*sample_[i4];
                  energy_difference_[i4] += diff_val_i4;
                  sum_val_i3 += diff_val_i4;
               }
               energy_difference_[i3] += sum_val_i3 + diff_val_i3;
               sum_val_i2 += sum_val_i3 + diff_val_i3;
            }
            energy_difference_[i2] += sum_val_i2 + diff_val_i2;
         }
      }
      else if (degree_ == 5) {
         const typename ModelType::ValueType target_ineraction_deg2 = interaction_[2];
         const typename ModelType::ValueType target_ineraction_deg3 = interaction_[3];
         const typename ModelType::ValueType target_ineraction_deg4 = interaction_[4];
         const typename ModelType::ValueType target_ineraction_deg5 = interaction_[5];
         const typename ModelType::OPType spin_prod1 = target_spin;
         for (std::int32_t i2 = 0; i2 < system_size_; ++i2) {
            if (i2 == index) {continue;}
            const typename ModelType::OPType spin_prod2 = spin_prod1*sample_[i2];
            const typename ModelType::ValueType diff_val_i2 = 4*target_ineraction_deg2*spin_prod2;
            typename ModelType::ValueType sum_val_i2 = 0;
            for (std::int32_t i3 = i2 + 1; i3 < system_size_; ++i3) {
               if (i3 == index) {continue;}
               const typename ModelType::OPType spin_prod3 = spin_prod2*sample_[i3];
               const typename ModelType::ValueType diff_val_i3 = 4*target_ineraction_deg3*spin_prod3;
               typename ModelType::ValueType sum_val_i3 = 0;
               for (std::int32_t i4 = i3 + 1; i4 < system_size_; ++i4) {
                  if (i4 == index) {continue;}
                  const typename ModelType::OPType spin_prod4 = spin_prod3*sample_[i4];
                  const typename ModelType::ValueType diff_val_i4 = 4*target_ineraction_deg4*spin_prod4;
                  typename ModelType::ValueType sum_val_i4 = 0;
                  for (std::int32_t i5 = i4 + 1; i5 < index; ++i5) {
                     const typename ModelType::ValueType diff_val_i5 = 4*target_ineraction_deg5*spin_prod4*sample_[i5];
                     energy_difference_[i5] += diff_val_i5;
                     sum_val_i4 += diff_val_i5;
                  }
                  for (std::int32_t i5 = std::max(index + 1, i4 + 1); i5 < system_size_; ++i5) {
                     const typename ModelType::ValueType diff_val_i5 = 4*target_ineraction_deg5*spin_prod4*sample_[i5];
                     energy_difference_[i5] += diff_val_i5;
                     sum_val_i4 += diff_val_i5;
                  }
                  energy_difference_[i4] += sum_val_i4 + diff_val_i4;
                  sum_val_i3 += sum_val_i4 + diff_val_i4;
               }
               energy_difference_[i3] += sum_val_i3 + diff_val_i3;
               sum_val_i2 += sum_val_i3 + diff_val_i3;
            }
            energy_difference_[i2] += sum_val_i2 + diff_val_i2;
         }
      }
      else if (degree_ == 6) {
         const typename ModelType::ValueType target_ineraction_deg2 = interaction_[2];
         const typename ModelType::ValueType target_ineraction_deg3 = interaction_[3];
         const typename ModelType::ValueType target_ineraction_deg4 = interaction_[4];
         const typename ModelType::ValueType target_ineraction_deg5 = interaction_[5];
         const typename ModelType::ValueType target_ineraction_deg6 = interaction_[6];
         const typename ModelType::OPType spin_prod1 = target_spin;
         for (std::int32_t i2 = 0; i2 < system_size_; ++i2) {
            if (i2 == index) {continue;}
            const typename ModelType::OPType spin_prod2 = spin_prod1*sample_[i2];
            const typename ModelType::ValueType diff_val_i2 = 4*target_ineraction_deg2*spin_prod2;
            typename ModelType::ValueType sum_val_i2 = 0;
            for (std::int32_t i3 = i2 + 1; i3 < system_size_; ++i3) {
               if (i3 == index) {continue;}
               const typename ModelType::OPType spin_prod3 = spin_prod2*sample_[i3];
               const typename ModelType::ValueType diff_val_i3 = 4*target_ineraction_deg3*spin_prod3;
               typename ModelType::ValueType sum_val_i3 = 0;
               for (std::int32_t i4 = i3 + 1; i4 < system_size_; ++i4) {
                  if (i4 == index) {continue;}
                  const typename ModelType::OPType spin_prod4 = spin_prod3*sample_[i4];
                  const typename ModelType::ValueType diff_val_i4 = 4*target_ineraction_deg4*spin_prod4;
                  typename ModelType::ValueType sum_val_i4 = 0;
                  for (std::int32_t i5 = i4 + 1; i5 < system_size_; ++i5) {
                     if (i5 == index) {continue;}
                     const typename ModelType::OPType spin_prod5 = spin_prod4*sample_[i5];
                     const typename ModelType::ValueType diff_val_i5 = 4*target_ineraction_deg5*spin_prod5;
                     typename ModelType::ValueType sum_val_i5 = 0;
                     for (std::int32_t i6 = i5 + 1; i6 < index; ++i6) {
                        const typename ModelType::ValueType diff_val_i6 = 4*target_ineraction_deg6*spin_prod5*sample_[i6];
                        energy_difference_[i6] += diff_val_i6;
                        sum_val_i5 += diff_val_i6;
                     }
                     for (std::int32_t i6 = std::max(index + 1, i5 + 1); i6 < system_size_; ++i6) {
                        const typename ModelType::ValueType diff_val_i6 = 4*target_ineraction_deg6*spin_prod5*sample_[i6];
                        energy_difference_[i6] += diff_val_i6;
                        sum_val_i5 += diff_val_i6;
                     }
                     energy_difference_[i5] += sum_val_i5 + diff_val_i5;
                     sum_val_i4 += sum_val_i5 + diff_val_i5;
                  }
                  energy_difference_[i4] += sum_val_i4 + diff_val_i4;
                  sum_val_i3 += sum_val_i4 + diff_val_i4;
               }
               energy_difference_[i3] += sum_val_i3 + diff_val_i3;
               sum_val_i2 += sum_val_i3 + diff_val_i3;
            }
            energy_difference_[i2] += sum_val_i2 + diff_val_i2;
         }
      }
      else if (degree_ == 7) {
         const typename ModelType::ValueType target_ineraction_deg2 = interaction_[2];
         const typename ModelType::ValueType target_ineraction_deg3 = interaction_[3];
         const typename ModelType::ValueType target_ineraction_deg4 = interaction_[4];
         const typename ModelType::ValueType target_ineraction_deg5 = interaction_[5];
         const typename ModelType::ValueType target_ineraction_deg6 = interaction_[6];
         const typename ModelType::ValueType target_ineraction_deg7 = interaction_[7];
         const typename ModelType::OPType spin_prod1 = target_spin;
         for (std::int32_t i2 = 0; i2 < system_size_; ++i2) {
            if (i2 == index) {continue;}
            const typename ModelType::OPType spin_prod2 = spin_prod1*sample_[i2];
            const typename ModelType::ValueType diff_val_i2 = 4*target_ineraction_deg2*spin_prod2;
            typename ModelType::ValueType sum_val_i2 = 0;
            for (std::int32_t i3 = i2 + 1; i3 < system_size_; ++i3) {
               if (i3 == index) {continue;}
               const typename ModelType::OPType spin_prod3 = spin_prod2*sample_[i3];
               const typename ModelType::ValueType diff_val_i3 = 4*target_ineraction_deg3*spin_prod3;
               typename ModelType::ValueType sum_val_i3 = 0;
               for (std::int32_t i4 = i3 + 1; i4 < system_size_; ++i4) {
                  if (i4 == index) {continue;}
                  const typename ModelType::OPType spin_prod4 = spin_prod3*sample_[i4];
                  const typename ModelType::ValueType diff_val_i4 = 4*target_ineraction_deg4*spin_prod4;
                  typename ModelType::ValueType sum_val_i4 = 0;
                  for (std::int32_t i5 = i4 + 1; i5 < system_size_; ++i5) {
                     if (i5 == index) {continue;}
                     const typename ModelType::OPType spin_prod5 = spin_prod4*sample_[i5];
                     const typename ModelType::ValueType diff_val_i5 = 4*target_ineraction_deg5*spin_prod5;
                     typename ModelType::ValueType sum_val_i5 = 0;
                     for (std::int32_t i6 = i5 + 1; i6 < system_size_; ++i6) {
                        if (i6 == index) {continue;}
                        const typename ModelType::OPType spin_prod6 = spin_prod5*sample_[i6];
                        const typename ModelType::ValueType diff_val_i6 = 4*target_ineraction_deg6*spin_prod6;
                        typename ModelType::ValueType sum_val_i6 = 0;
                        for (std::int32_t i7 = i6 + 1; i7 < index; ++i7) {
                           const typename ModelType::ValueType diff_val_i7 = 4*target_ineraction_deg7*spin_prod6*sample_[i7];
                           energy_difference_[i7] += diff_val_i7;
                           sum_val_i6 += diff_val_i7;
                        }
                        for (std::int32_t i7 = std::max(index + 1, i6 + 1); i7 < system_size_; ++i7) {
                           const typename ModelType::ValueType diff_val_i7 = 4*target_ineraction_deg7*spin_prod6*sample_[i7];
                           energy_difference_[i7] += diff_val_i7;
                           sum_val_i6 += diff_val_i7;
                        }
                        energy_difference_[i6] += sum_val_i6 + diff_val_i6;
                        sum_val_i5 += sum_val_i6 + diff_val_i6;
                     }
                     energy_difference_[i5] += sum_val_i5 + diff_val_i5;
                     sum_val_i4 += sum_val_i5 + diff_val_i5;
                  }
                  energy_difference_[i4] += sum_val_i4 + diff_val_i4;
                  sum_val_i3 += sum_val_i4 + diff_val_i4;
               }
               energy_difference_[i3] += sum_val_i3 + diff_val_i3;
               sum_val_i2 += sum_val_i3 + diff_val_i3;
            }
            energy_difference_[i2] += sum_val_i2 + diff_val_i2;
         }
      }
      else {
         for (std::int32_t p = 2; p < static_cast<std::int32_t>(interaction_.size()); ++p) {
            if (std::abs(interaction_[p]) > std::numeric_limits<typename ModelType::ValueType>::epsilon()) {
               const typename ModelType::ValueType target_ineraction = interaction_[p];
               std::vector<std::int32_t> indices(p);
               std::int32_t start_index = 0;
               std::int32_t size = 0;
               
               while (true) {
                  for (std::int32_t i = start_index; i < system_size_ - 1; ++i) {
                     indices[size++] = i;
                     if (size == p) {
                        typename ModelType::OPType sign = 1;
                        for (std::int32_t j = 0; j < p; ++j) {
                           if (indices[j] >= index) {
                              sign *= sample_[indices[j] + 1];
                           }
                           else {
                              sign *= sample_[indices[j]];
                           }
                        }
                        const typename ModelType::ValueType val = 4*target_spin*sign*target_ineraction;
                        for (std::int32_t j = 0; j < p; ++j) {
                           if (indices[j] >= index) {
                              energy_difference_[indices[j] + 1] += val;
                           }
                           else {
                              energy_difference_[indices[j]] += val;
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
         }
      }
      
      energy_difference_[index] *= -1;
      sample_[index] *= -1;
   }
   
   const std::vector<typename ModelType::OPType> &GetSample() const {
      return sample_;
   }
   
   typename ModelType::ValueType GetEnergyDifference(const std::int32_t index) const {
      return energy_difference_[index];
   }
   
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
private:
   const std::int32_t system_size_;
   const lattice::BoundaryCondition bc_;
   const std::vector<typename ModelType::ValueType> interaction_;
   const std::int32_t degree_;
   
   std::vector<typename ModelType::OPType> sample_;
   std::vector<typename ModelType::ValueType> energy_difference_;
   
   std::vector<typename ModelType::ValueType> GenerateEnergyDifference(const std::vector<typename ModelType::OPType> &sample) const {
      std::vector<typename ModelType::ValueType> energy_difference(system_size_);
      for (std::int32_t index = 0; index < system_size_; ++index) {
         typename ModelType::ValueType val = 0.0;
         const typename ModelType::OPType target_spin = sample[index];
      
         if (interaction_.size() >= 2) {
            val += interaction_[1]*target_spin;
         }
         for (std::int32_t p = 2; p < static_cast<std::int32_t>(interaction_.size()); ++p) {
            if (std::abs(interaction_[p]) > std::numeric_limits<typename ModelType::ValueType>::epsilon()) {
               const typename ModelType::ValueType target_ineraction = interaction_[p];
               std::vector<std::int32_t> indices(p - 1);
               std::int32_t start_index = 0;
               std::int32_t size = 0;
      
               while (true) {
                  for (std::int32_t i = start_index; i < system_size_ - 1; ++i) {
                     indices[size++] = i;
                     if (size == p - 1) {
                        typename ModelType::OPType sign = 1;
                        for (std::int32_t j = 0; j < p - 1; ++j) {
                           if (indices[j] >= index) {
                              sign *= sample[indices[j] + 1];
                           }
                           else {
                              sign *= sample[indices[j]];
                           }
                        }
                        val += target_ineraction*sign*target_spin;
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
         }
         energy_difference[index] = -2.0*val;
      }
      return energy_difference;
   }

};

template<typename RealType>
CMCSystem(const model::PolynomialIsing<lattice::InfiniteRange, RealType>) -> CMCSystem<model::PolynomialIsing<lattice::InfiniteRange, RealType>>;

} // namespace cmc_utility
} // namespace solver
} // namespace compnal



#endif /* COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_POLY_ISING_INFINITE_RANGE_HPP_ */
