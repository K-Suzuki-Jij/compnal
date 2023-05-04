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
//  classical_monte_carlo.hpp
//  compnal
//
//  Created by kohei on 2023/05/04.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_

#include "../parameter_class.hpp"
#include <random>

namespace compnal {
namespace solver {

template<class ModelType>
class ClassicalMonteCarlo {
   
   //! @brief The operator type
   using PHQType = typename ModelType::PHQType;
   
public:
   //! @brief Constructor for ClassicalMonteCarlo class.
   //! @param model The classical model.
   ClassicalMonteCarlo(const ModelType &model): model_(model) {}
   
   //! @brief Set the number of sweeps.
   //! @param num_sweeps The number of sweeps, which must be non-negative integer.
   void SetNumSweeps(const std::int32_t num_sweeps) {
      if (num_sweeps < 0) {
         throw std::invalid_argument("num_sweeps must be non-negative integer.");
      }
      num_sweeps_ = num_sweeps;
   }
   
   //! @brief Set the number of samples.
   //! @param num_samples The number of samples, which must be non-negative integer.
   void SetNumSamples(const std::int32_t num_samples) {
      if (num_samples <= 0) {
         throw std::invalid_argument("num_samples must be positive integer.");
      }
      num_samples_ = num_samples;
   }
   
   //! @brief Set the number of threads in the calculation.
   //! @param num_threads The number of threads in the calculation, which must be larger than zero.
   void SetNumThreads(const std::int32_t num_threads) {
      if (num_threads <= 0) {
         throw std::invalid_argument("num_threads must be non-negative integer.");
      }
      num_threads_ = num_threads;
   }
   
   //! @brief Set the temperature.
   //! @param temperature The temperature, which must be larger than zero.
   void SetTemperature(const double temperature) {
      if (temperature < 0) {
         throw std::invalid_argument("Temperature must be non-negative value.");
      }
      temperature_ = 1.0/temperature;
   }
   
   //! @brief Set update method used in the state update.
   //! @param updater The update method.
   void SetUpdater(const MonteCarloUpdater updater) {
      updater_ = updater;
   }
   
   //! @brief Set random number engine for updating initializing state.
   //! @param random_number_engine The random number engine.
   void SetRandomNumberEngine(const RandomNumberEngine random_number_engine) {
      random_number_engine_ = random_number_engine;
   }
   
private:
   //! @brief The model.
   const ModelType model_;
   
   //! @brief The number of sweeps.
   std::int32_t num_sweeps_ = 1000;
   
   //! @brief The number of samples.
   std::int32_t num_samples_ = 1;
   
   //! @brief The number of threads in the calculation.
   std::int32_t num_threads_ = 1;
   
   //! @brief The temperature.
   double temperature_ = 1;
   
   //! @brief The seed to be used in the calculation.
   std::uint64_t seed_ = std::random_device()();
   
   //! @brief The samples.
   std::vector<std::vector<PHQType>> samples_ = std::vector<std::vector<PHQType>>(num_samples_);
   
   //! @brief State updater.
   MonteCarloUpdater updater_ = MonteCarloUpdater::METROPOLIS;
   
   //! @brief Random number engine.
   RandomNumberEngine random_number_engine_ = RandomNumberEngine::MT;
   
};


} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
