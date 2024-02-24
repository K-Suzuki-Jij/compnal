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
//  variable.hpp
//  compnal
//
//  Created by kohei on 2023/05/07.
//  
//

#pragma once

#include <stdexcept>
#include <cmath>
#include <sstream>

namespace compnal {
namespace model {
namespace utility {

//! @brief Class for representing spin variables.
//! \f[ s_{i}\in\{-s,-s+1,\ldots, s-1,s\} \f]
class Spin {
   
public:
   //! @brief Constructor for Spin class.
   //! @param spin_magnitude The magnitude of spin.
   //! @param spin_scale_factor A scaling factor used to adjust the value taken by the spin.
   //! The default value is 1.0, which represents the usual spin, taking value \f[ s\in\{-1/2,+1/2\} \f].
   //! By changing this value, you can represent spins of different values,
   //! such as \f[ s\in\{-1,+1\} \f] by setting spin_scale_factor=2.
   Spin(const double spin_magnitude, const std::int32_t spin_scale_factor):
   state_number_(0),
   twice_spin_magnitude_(static_cast<std::int32_t>(2*spin_magnitude)),
   spin_scale_factor_(spin_scale_factor),
   num_state_(twice_spin_magnitude_ + 1) {
      if (std::floor(2*spin_magnitude) != 2*spin_magnitude || spin_magnitude <= 0) {
         throw std::invalid_argument("spin_magnitude must be positive half-integer.");
      }
      if (spin_scale_factor < 1) {
         throw std::invalid_argument("spin_scale_factor must positive-integer");
      }
   }
   
   //! @brief Get the number of state.
   //! @return The number of state.
   std::int32_t GetNumState() const {
      return num_state_;
   }
      
   //! @brief Get the value of the spin variable from the state number.
   //! @param state_number The state number of the spin variable.
   //! @return The value of the spin variable.
   double GetValueFromState(const std::int32_t state_number) const {
      if (state_number < 0 || state_number >= num_state_) {
         std::stringstream ss;
         ss << "state_number must be in [0," << num_state_ << ")." << std::endl;
         ss << "But state_number=" << state_number;
         throw std::invalid_argument(ss.str());
      }
      return (-1*twice_spin_magnitude_*0.5 + state_number)*spin_scale_factor_;
   }
   
   //! @brief Get the value.
   //! @return The value of the spin variable.
   double GetValue() const {
      return (-1*twice_spin_magnitude_*0.5 + state_number_)*spin_scale_factor_;
   }
   
   //! @brief Get the state number.
   //! @return The state number.
   std::int32_t GetStateNumber() const {
      return state_number_;
   }
   
   //! @brief Generate candidate state.
   //! @tparam RandType Random number engine type.
   //! @param random_number_engine Random number engine.
   //! @return The candidate state number.
   template<class RandType>
   std::int32_t GenerateCandidateState(RandType *random_number_engine) const {
      if (twice_spin_magnitude_ == 1) {
         return 1 - state_number_;
      }
      else {
         std::int32_t new_state_number = (*random_number_engine)()%(num_state_ - 1);
         if (state_number_ <= new_state_number) {
            new_state_number++;
         }
         return new_state_number;
      }
   }
   
   //! @brief Set the state number.
   //! @param state_number The state number of the spin variable.
   void SetState(const std::int32_t state_number) {
      if (state_number < 0 || state_number >= num_state_) {
         std::stringstream ss;
         ss << "state_number must be in [0," << num_state_ << ")." << std::endl;
         ss << "But state_number=" << state_number;
         throw std::invalid_argument(ss.str());
      }
      state_number_ = state_number;
   }
   
   void SetValue(const double value) {
      // Calculate value to the state number
      const double new_state_number = (value/spin_scale_factor_ + twice_spin_magnitude_*0.5);
      
      // Check if new_state_number is integer
      if (std::floor(new_state_number) != new_state_number) {
         std::stringstream ss;
         ss << "value must be integer multiple of spin_scale_factor." << std::endl;
         ss << "But value=" << value << ", spin_scale_factor=" << spin_scale_factor_;
         throw std::invalid_argument(ss.str());
      }

      // Set state number
      SetState(static_cast<std::int32_t>(new_state_number));
   }
   
   //! @brief Set random state.
   //! @tparam RandType Random number engine type.
   //! @param random_number_engine Random number engine.
   template<class RandType>
   void SetStateRandomly(RandType *random_number_engine) {
      state_number_ = (*random_number_engine)()%num_state_;
   }
   
   
private:
   //! @brief The state number of the spin variable.
   std::int32_t state_number_ = 0;

   //! @brief Twice magnitude of spin.
   const std::int32_t twice_spin_magnitude_ = 1;

   //! @brief spin_scale_factor A scaling factor used to adjust the value taken by the spin.
   //! The default value is 1.0, which represents the usual spin, taking value \f[ s\in\{-1/2,+1/2\} \f].
   //! By changing this value, you can represent spins of different values,
   //! such as \f[ s\in\{-1,+1\} \f] by setting spin_scale_factor=2.
   const std::int32_t spin_scale_factor_ = 1;

   //! @brief The number of candidate states.
   const std::int32_t num_state_ = twice_spin_magnitude_ + 1;
   
};


}
} // namespace model
} // namespace compnal

