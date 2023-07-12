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
#include "parallel_tempering.hpp"
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
   
   //! @brief Calculate energies for each sample.
   //! @return The list of energy.
   Eigen::Vector<PHQType, Eigen::Dynamic>
   CalculateEnergies(const ModelType &model,
                     const Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &samples,
                     const std::int32_t num_threads) const {
      const std::int32_t num_samples = static_cast<std::int32_t>(samples.rows());
      Eigen::Vector<PHQType, Eigen::Dynamic> energies(num_samples);
      
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::int32_t i = 0; i < num_samples; ++i) {
         energies(i) = model.CalculateEnergy(samples.row(i));
      }
      return energies;
   }
   
   //! @brief Execute classical monte carlo simulation.
   //! @param seed The seed used in the calculation.
   Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
   RunSingleFlip(const ModelType &model,
                 const std::int32_t num_sweeps,
                 const std::int32_t num_samples,
                 const std::int32_t num_threads,
                 const double temperature,
                 const std::uint64_t seed,
                 const StateUpdateMethod updater,
                 const RandomNumberEngine random_number_engine,
                 const SpinSelectionMethod spin_selector) const {
      
      if (num_sweeps < 0) {
         throw std::invalid_argument("num_sweeps must be non-negative integer.");
      }
      if (num_samples <= 0) {
         throw std::invalid_argument("num_samples must be positive integer.");
      }
      if (num_threads <= 0) {
         throw std::invalid_argument("num_threads must be non-negative integer.");
      }
      if (temperature < 0) {
         throw std::invalid_argument("Temperature must be non-negative value.");
      }
      
      if (random_number_engine == RandomNumberEngine::XORSHIFT) {
         return TemplateSingleUpdater<System<ModelType, utility::Xorshift>, utility::Xorshift>
         (model, num_sweeps, num_samples, num_threads, temperature, seed,
          updater, random_number_engine, spin_selector);
      }
      else if (random_number_engine == RandomNumberEngine::MT) {
         return TemplateSingleUpdater<System<ModelType, std::mt19937>, std::mt19937>
         (model, num_sweeps, num_samples, num_threads, temperature, seed,
          updater, random_number_engine, spin_selector);
      }
      else if (random_number_engine == RandomNumberEngine::MT_64) {
         return TemplateSingleUpdater<System<ModelType, std::mt19937_64>, std::mt19937_64>
         (model, num_sweeps, num_samples, num_threads, temperature, seed,
          updater, random_number_engine, spin_selector);
      }
      else {
         throw std::invalid_argument("Unknwon random_number_engine");
      }
   }
   
   //! @brief Execute classical monte carlo simulation.
   //! @param seed The seed used in the calculation.
   Eigen::Vector<Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Dynamic>
   RunParallelTempering(const ModelType &model,
                        const std::int32_t num_sweeps,
                        const std::int32_t num_swaps,
                        const std::int32_t num_replica,
                        const std::int32_t num_samples,
                        const std::int32_t num_threads,
                        const std::pair<double, double> temperature_range,
                        const std::uint64_t seed,
                        const TemperatureDistribution temperature_distribution,
                        const StateUpdateMethod updater,
                        const RandomNumberEngine random_number_engine,
                        const SpinSelectionMethod spin_selector) const {
      
      if (num_sweeps < 0) {
         throw std::invalid_argument("num_sweeps must be non-negative integer.");
      }
      if (num_swaps < 0) {
         throw std::invalid_argument("num_swaps must be non-negative integer.");
      }
      if (num_replica < 0) {
         throw std::invalid_argument("num_replica must be non-negative integer.");
      }
      if (num_samples <= 0) {
         throw std::invalid_argument("num_samples must be positive integer.");
      }
      if (num_threads <= 0) {
         throw std::invalid_argument("num_threads must be non-negative integer.");
      }
      if (temperature_range.first < 0 || temperature_range.second < 0) {
         throw std::invalid_argument("Temperature must be non-negative value.");
      }
      if (temperature_range.first > temperature_range.second) {
         throw std::invalid_argument("Temperature range is invalid");
      }
      
      if (random_number_engine == RandomNumberEngine::XORSHIFT) {
         return TemplateParallelTempering<System<ModelType, utility::Xorshift>, utility::Xorshift>
         (model, num_sweeps, num_swaps, num_replica, num_samples, num_threads,
          temperature_range, seed, temperature_distribution, updater, random_number_engine, spin_selector);
      }
      else if (random_number_engine == RandomNumberEngine::MT) {
         return TemplateParallelTempering<System<ModelType, std::mt19937>, std::mt19937>
         (model, num_sweeps, num_swaps, num_replica, num_samples, num_threads,
          temperature_range, seed, temperature_distribution, updater, random_number_engine, spin_selector);
      }
      else if (random_number_engine == RandomNumberEngine::MT_64) {
         return TemplateParallelTempering<System<ModelType, std::mt19937_64>, std::mt19937_64>
         (model, num_sweeps, num_swaps, num_replica, num_samples, num_threads,
          temperature_range, seed, temperature_distribution, updater, random_number_engine, spin_selector);
      }
      else {
         throw std::invalid_argument("Unknwon random_number_engine");
      }
      
   }
   
private:
   //! @brief Template function for running classical monte carlo simulation.
   //! @tparam SystemType System class.
   //! @tparam RandType Random number engine class.
   //! @param seed The seed used in the calculation.
   //! @return The list of samples.
   template<class SystemType, class RandType>
   Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
   TemplateSingleUpdater(const ModelType &model,
                         const std::int32_t num_sweeps,
                         const std::int32_t num_samples,
                         const std::int32_t num_threads,
                         const double temperature,
                         const std::uint64_t seed,
                         const StateUpdateMethod updater,
                         const RandomNumberEngine random_number_engine,
                         const SpinSelectionMethod spin_selector) const {
      
      using RType = typename RandType::result_type;
      using E2DType = Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      
      E2DType samples(num_samples, model.GetLattice().GetSystemSize());
      std::vector<RType> system_seed(num_samples);
      std::vector<RType> updater_seed(num_samples);
      RandType rand(static_cast<RType>(seed));
      for (std::int32_t i = 0; i < num_samples; ++i) {
         system_seed[i] = rand();
         updater_seed[i] = rand();
      }
      
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::int32_t i = 0; i < num_samples; ++i) {
         auto system = SystemType{model, system_seed[i]};
         SingleUpdater<SystemType, RandType>(&system, num_sweeps, 1.0/temperature, updater_seed[i], updater, spin_selector);
         samples.row(i) = system.ExtractSample();
      }
      return samples;
   }
   
   //! @brief Template function for running classical monte carlo simulation.
   //! @tparam SystemType System class.
   //! @tparam RandType Random number engine class.
   //! @param seed The seed used in the calculation.
   //! @return The list of samples.
   template<class SystemType, class RandType>
   Eigen::Vector<Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Dynamic>
   TemplateParallelTempering(const ModelType &model,
                             const std::int32_t num_sweeps,
                             const std::int32_t num_swaps,
                             const std::int32_t num_replica,
                             const std::int32_t num_samples,
                             const std::int32_t num_threads,
                             const std::pair<double, double> temperature_range,
                             const std::uint64_t seed,
                             const TemperatureDistribution temperature_distribution,
                             const StateUpdateMethod updater,
                             const RandomNumberEngine random_number_engine,
                             const SpinSelectionMethod spin_selector) const {
      
      using RType = typename RandType::result_type;
      using E2DType = Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      
      const auto beta_list = GenerateBetaList(temperature_range, num_replica, temperature_distribution);
      Eigen::Vector<E2DType, Eigen::Dynamic> samples(num_samples);
      std::vector<std::vector<RType>> system_seed(num_samples, std::vector<RType>(num_replica));
      std::vector<RType> updater_seed(num_samples);
      RandType rand(static_cast<RType>(seed));
      for (std::int32_t i = 0; i < num_samples; ++i) {
         updater_seed[i] = rand();
         for (std::int32_t j = 0; j < num_replica; ++j) {
            system_seed[i][j] = rand();
         }
      }
     
      for (std::int32_t i = 0; i < num_samples; ++i) {
         std::vector<SystemType> system_list;
         for (std::int32_t j = 0; j < num_replica; ++j) {
            system_list.push_back(SystemType{model, system_seed[i][j]});
         }
         
         
         std::vector<SystemType*> system_list_pointer;
         for (std::int32_t j = 0; j < num_replica; ++j) {
            system_list_pointer.push_back(&system_list[j]);
         }
         
         ParallelTempering<SystemType, RandType>(&system_list_pointer,
                                                 num_sweeps, num_swaps, num_threads,
                                                 updater_seed[i], beta_list,
                                                 updater, spin_selector);
         
         E2DType temp_samples(num_replica, model.GetLattice().GetSystemSize());
         
         for (std::int32_t j = 0; j < num_replica; ++j) {
            temp_samples.row(j) = system_list[j].ExtractSample();
         }
         samples(i) = temp_samples;
      }
      
      
      
      return samples;
   }
};

//! @brief Make ClassicalMonteCarlo class.
//! @tparam ModelType Model class.
//! @param model The model.
//! @return The ClassicalMonteCarlo class.
template<class ModelType>
auto make_classical_monte_carlo() {
   return ClassicalMonteCarlo<ModelType>();
}

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
