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

#include "../../utility/random.hpp"
#include "../parameter_class.hpp"
#include "system/all.hpp"
#include "single_updater.hpp"
#include <random>
#include <Eigen/Dense>

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Classical monte carlo class.
//! @tparam ModelType Model class.
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
      temperature_ = temperature;
   }
   
   //! @brief Set update method used in the state update.
   //! @param updater The update method.
   void SetStateUpdateMethod(const StateUpdateMethod updater) {
      updater_ = updater;
   }
   
   //! @brief Set random number engine for updating initializing state.
   //! @param random_number_engine The random number engine.
   void SetRandomNumberEngine(const RandomNumberEngine random_number_engine) {
      random_number_engine_ = random_number_engine;
   }
   
   //! @brief Set spin selection method.
   void SetSpinSelectionMethod(const SpinSelectionMethod spin_selection_method) {
      spin_selector_ = spin_selection_method;
   }
   
   //! @brief Get the number of sweeps.
   //! @return The number of sweeps.
   std::int32_t GetNumSweeps() const {
      return num_sweeps_;
   }
   
   //! @brief Get the number of samples.
   //! @return The number of samples.
   std::int32_t GetNumSamples() const {
      return num_samples_;
   }
   
   //! @brief Get the number of threads in the calculation.
   //! @return The number of threads in the calculation.
   std::int32_t GetNumThreads() const {
      return num_threads_;
   }
   
   //! @brief Get the temperature.
   //! @return The temperature.
   double GetTemperature() const {
      return temperature_;
   }
   
   //! @brief Get the state update method.
   //! @return The StateUpdateMethod.
   StateUpdateMethod GetStateUpdateMethod() const {
      return updater_;
   }
   
   //! @brief Get the random number engine.
   //! @return The random number engine.
   RandomNumberEngine GetRandomNumberEngine() const {
      return random_number_engine_;
   }
   
   //! @brief Get the spin selection method.
   //! @return The spin selection method.
   SpinSelectionMethod GetSpinSelectionMethod() const {
      return spin_selector_;
   }
   
   //! @brief Get the seed to be used in the calculation.
   //! @return The seed.
   std::uint64_t GetSeed() const {
      return seed_;
   }
   
   //! @brief Get the model.
   //! @return The model.
   const ModelType GetModel() const {
      return model_;
   }
   
   //! @brief Get the list of samples.
   //! @return The list of samples.
   const Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &GetSamples() const {
      return samples_;
   }
   
   //! @brief Calculate energies for each sample.
   //! @return The list of energy.
   Eigen::Vector<PHQType, Eigen::Dynamic> CalculateEnergies() const {
      Eigen::Vector<PHQType, Eigen::Dynamic> energies(num_samples_);
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int32_t i = 0; i < num_samples_; ++i) {
         energies(i) = model_.CalculateEnergy(samples_.row(i));
      }
      return energies;
   }
   
   //! @brief Execute classical monte carlo simulation.
   void RunSampling() {
      RunSampling(std::random_device()());
   }
   
   //! @brief Execute classical monte carlo simulation.
   //! @param seed The seed used in the calculation.
   void RunSampling(const std::uint64_t seed) {
      if (random_number_engine_ == RandomNumberEngine::XORSHIFT) {
         samples_ = TemplateRunner<System<ModelType, utility::Xorshift>, utility::Xorshift>(static_cast<std::uint32_t>(seed));
      }
      else if (random_number_engine_ == RandomNumberEngine::MT) {
         samples_ = TemplateRunner<System<ModelType, std::mt19937>, std::mt19937>(static_cast<std::uint32_t>(seed));
      }
      else if (random_number_engine_ == RandomNumberEngine::MT_64) {
         samples_ = TemplateRunner<System<ModelType, std::mt19937_64>, std::mt19937_64>(seed);
      }
      else {
         throw std::invalid_argument("Unknwon random_number_engine");
      }
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
   
   //! @brief The seed used in the calculation.
   std::uint64_t seed_ = std::random_device()();
   
   //! @brief The list of samples.
   Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> samples_;
   
   //! @brief State updater.
   StateUpdateMethod updater_ = StateUpdateMethod::METROPOLIS;
   
   //! @brief Random number engine.
   RandomNumberEngine random_number_engine_ = RandomNumberEngine::MT;
   
   //! @brief Spin selective method.
   SpinSelectionMethod spin_selector_ = SpinSelectionMethod::RANDOM;
   
   //! @brief Template function for running classical monte carlo simulation.
   //! @tparam SystemType System class.
   //! @tparam RandType Random number engine class.
   //! @param seed The seed used in the calculation.
   //! @return The list of samples.
   template<class SystemType, class RandType>
   Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TemplateRunner(const typename RandType::result_type seed) {
      seed_ = seed;
      Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> samples(num_samples_, model_.GetLattice().GetSystemSize());
      std::vector<typename RandType::result_type> system_seed(num_samples_);
      std::vector<typename RandType::result_type> updater_seed(num_samples_);
      RandType random_number_engine(seed);
      for (std::int32_t i = 0; i < num_samples_; ++i) {
         system_seed[i] = random_number_engine();
         updater_seed[i] = random_number_engine();
      }
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int32_t i = 0; i < num_samples_; ++i) {
         auto system = SystemType{model_, system_seed[i]};
         SingleUpdater<SystemType, RandType>(&system, num_sweeps_, 1.0/temperature_, updater_seed[i], updater_, spin_selector_);
         samples.row(i) = system.ExtractSample();
      }
      return samples;
   }
   
};

//! @brief Make ClassicalMonteCarlo class.
//! @tparam ModelType Model class.
//! @param model The model.
//! @return The ClassicalMonteCarlo class.
template<class ModelType>
auto make_classical_monte_carlo(const ModelType &model) {
   return ClassicalMonteCarlo<ModelType>(model);
}

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
