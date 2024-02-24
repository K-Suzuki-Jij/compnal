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
//  classical_monte_carlo.hpp
//  compnal
//
//  Created by kohei on 2023/05/04.
//  
//

#pragma once

#include "../../utility/random.hpp"
#include "../parameter_class.hpp"
#include "system/all.hpp"
#include "metropolis_ssf.hpp"
#include "metropolis_pt.hpp"
#include "heat_bath_ssf.hpp"
#include "heat_bath_pt.hpp"
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
   //! @param model The model.
   //! @param samples The list of samples.
   //! @param num_threads The number of calculation threads.
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
   //! @param model The model.
   //! @param num_sweeps The number of sweeps.
   //! @param num_samples The number of samples.
   //! @param num_threads The number of calculation threads.
   //! @param temperature The temperature.
   //! @param seed The seed used in the calculation.
   //! @param updater The state update method.
   //! @param random_number_engine The random number engine.
   //! @param spin_selector The spin selection method.
   //! @return The samples.
   Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
   RunSingleFlip(const ModelType &model,
                 const std::int32_t num_sweeps,
                 const std::int32_t num_samples,
                 const std::int32_t num_threads,
                 const double temperature,
                 const std::uint64_t seed,
                 const StateUpdateMethod updater,
                 const RandomNumberEngine random_number_engine,
                 const SpinSelectionMethod spin_selector,
                 const Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &initial_sample_list = {}) const {
      
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
          updater, random_number_engine, spin_selector, initial_sample_list);
      }
      else if (random_number_engine == RandomNumberEngine::MT) {
         return TemplateSingleUpdater<System<ModelType, std::mt19937>, std::mt19937>
         (model, num_sweeps, num_samples, num_threads, temperature, seed,
          updater, random_number_engine, spin_selector, initial_sample_list);
      }
      else if (random_number_engine == RandomNumberEngine::MT_64) {
         return TemplateSingleUpdater<System<ModelType, std::mt19937_64>, std::mt19937_64>
         (model, num_sweeps, num_samples, num_threads, temperature, seed,
          updater, random_number_engine, spin_selector, initial_sample_list);
      }
      else {
         throw std::invalid_argument("Unknwon random_number_engine");
      }
   }
   
   //! @brief Execute classical monte carlo simulation.
   //! @param model The model.
   //! @param num_sweeps The number of sweeps.
   //! @param num_swaps The number of swaps for each replica.
   //! @param num_samples The number of samples.
   //! @param num_threads The number of calculation threads.
   //! @param temperature_list The list of temperature.
   //! @param seed The seed used in the calculation.
   //! @param updater The state update method.
   //! @param random_number_engine The random number engine.
   //! @param spin_selector The spin selection method.
   //! @return The samples for each temperature.
   Eigen::Vector<PHQType, Eigen::Dynamic>
   RunParallelTempering(const ModelType &model,
                        const std::int32_t num_sweeps,
                        const std::int32_t num_swaps,
                        const std::int32_t num_samples,
                        const std::int32_t num_threads,
                        const Eigen::Vector<double, Eigen::Dynamic> &temperature_list,
                        const std::uint64_t seed,
                        const StateUpdateMethod updater,
                        const RandomNumberEngine random_number_engine,
                        const SpinSelectionMethod spin_selector) const {
      
      if (num_sweeps < 0) {
         throw std::invalid_argument("num_sweeps must be non-negative integer.");
      }
      if (num_swaps < 0) {
         throw std::invalid_argument("num_swaps must be non-negative integer.");
      }
      if (num_samples <= 0) {
         throw std::invalid_argument("num_samples must be positive integer.");
      }
      if (num_threads <= 0) {
         throw std::invalid_argument("num_threads must be non-negative integer.");
      }
      if (temperature_list.size() == 0) {
         throw std::invalid_argument("Thhe size of temperature_list must be larger than 0.");
      }
      
      if (random_number_engine == RandomNumberEngine::XORSHIFT) {
         return TemplateParallelTempering<System<ModelType, utility::Xorshift>, utility::Xorshift>
         (model, num_sweeps, num_swaps, num_samples, num_threads,
          temperature_list, seed, updater, random_number_engine, spin_selector);
      }
      else if (random_number_engine == RandomNumberEngine::MT) {
         return TemplateParallelTempering<System<ModelType, std::mt19937>, std::mt19937>
         (model, num_sweeps, num_swaps, num_samples, num_threads,
          temperature_list, seed, updater, random_number_engine, spin_selector);
      }
      else if (random_number_engine == RandomNumberEngine::MT_64) {
         return TemplateParallelTempering<System<ModelType, std::mt19937_64>, std::mt19937_64>
         (model, num_sweeps, num_swaps, num_samples, num_threads,
          temperature_list, seed, updater, random_number_engine, spin_selector);
      }
      else {
         throw std::invalid_argument("Unknwon random_number_engine");
      }
      
   }
   
private:
   //! @brief Template function for running classical monte carlo simulation.
   //! @tparam SystemType System class.
   //! @tparam RandType Random number engine class.
   //! @param model The model.
   //! @param num_sweeps The number of sweeps.
   //! @param num_samples The number of samples.
   //! @param num_threads The number of calculation threads.
   //! @param temperature The temperature.
   //! @param seed The seed used in the calculation.
   //! @param updater The state update method.
   //! @param random_number_engine The random number engine.
   //! @param spin_selector The spin selection method.
   //! @return The samples.
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
                         const SpinSelectionMethod spin_selector,
                         const Eigen::Matrix<PHQType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &initial_sample_list = {}) const {
      
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

      if (initial_sample_list.size() != 0 && initial_sample_list.rows() != num_samples) {
         throw std::invalid_argument("The size of initial_sample_list must be equal to num_samples.");
      }
      
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::int32_t i = 0; i < num_samples; ++i) {
         auto system = SystemType{model, system_seed[i]};
         if (initial_sample_list.size() != 0) {
            system.SetSampleByValue(initial_sample_list.row(i));
         }
         if (updater == StateUpdateMethod::METROPOLIS) {
            MetropolisSSF<SystemType, RandType>(&system, num_sweeps, 1.0/temperature, updater_seed[i], spin_selector);
         }
         else if (updater == StateUpdateMethod::HEAT_BATH) {
            HeatBathSSF<SystemType, RandType>(&system, num_sweeps, 1.0/temperature, updater_seed[i], spin_selector);
         }
         else {
            throw std::invalid_argument("Unknown updater");
         }
         
         samples.row(i) = system.ExtractSample();
      }
      return samples;
   }
   
   //! @brief Template function for running classical monte carlo simulation.
   //! @tparam SystemType System class.
   //! @tparam RandType Random number engine class.
   //! @param model The model.
   //! @param num_sweeps The number of sweeps.
   //! @param num_swaps The number of swaps for each replica.
   //! @param num_samples The number of samples.
   //! @param num_threads The number of calculation threads.
   //! @param temperature_list The list of temperatures.
   //! @param seed The seed used in the calculation.
   //! @param updater The state update method.
   //! @param random_number_engine The random number engine.
   //! @param spin_selector The spin selection method.
   //! @return The samples for each temperature.
   template<class SystemType, class RandType>
   Eigen::Vector<PHQType, Eigen::Dynamic>
   TemplateParallelTempering(const ModelType &model,
                             const std::int32_t num_sweeps,
                             const std::int32_t num_swaps,
                             const std::int32_t num_samples,
                             const std::int32_t num_threads,
                             const Eigen::Vector<double, Eigen::Dynamic> &temperature_list,
                             const std::uint64_t seed,
                             const StateUpdateMethod updater,
                             const RandomNumberEngine random_number_engine,
                             const SpinSelectionMethod spin_selector) const {
      
      using RType = typename RandType::result_type;
      
      const std::int64_t system_size = model.GetLattice().GetSystemSize();
      const std::int64_t num_replicas = static_cast<std::int64_t>(temperature_list.size());
      std::vector<double> beta_list(num_replicas);
      for (std::int64_t i = 0; i < num_replicas; ++i) {
         beta_list[i] = 1.0/temperature_list[i];
      }
            
      Eigen::Vector<PHQType, Eigen::Dynamic> samples(num_replicas*num_samples*system_size);
      std::vector<std::vector<RType>> system_seed(num_samples, std::vector<RType>(num_replicas));
      std::vector<RType> updater_seed(num_samples);
      RandType rand(static_cast<RType>(seed));
      
      for (std::int64_t i = 0; i < num_samples; ++i) {
         updater_seed[i] = rand();
         for (std::int64_t j = 0; j < num_replicas; ++j) {
            system_seed[i][j] = rand();
         }
      }
            
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::int64_t sample_count = 0; sample_count < num_samples; ++sample_count) {
         std::vector<SystemType> system_list;
         system_list.reserve(num_replicas);
         for (std::int64_t j = 0; j < num_replicas; ++j) {
            system_list.push_back(SystemType{model, system_seed[sample_count][j]});
         }
         
         std::vector<SystemType*> system_list_pointer;
         system_list_pointer.reserve(num_replicas);
         for (std::int64_t j = 0; j < num_replicas; ++j) {
            system_list_pointer.push_back(&system_list[j]);
         }
         
         if (updater == StateUpdateMethod::METROPOLIS) {
            MetropolisPT<SystemType, RandType>(&system_list_pointer,
                                               num_sweeps, num_swaps,
                                               updater_seed[sample_count], 
                                               beta_list, spin_selector);
         }
         else if (updater == StateUpdateMethod::HEAT_BATH) {
            HeatBathPT<SystemType, RandType>(&system_list_pointer,
                                             num_sweeps, num_swaps,
                                             updater_seed[sample_count],
                                             beta_list, spin_selector);
         }
         else {
            throw std::invalid_argument("Unknown updater");
         }
         
         for (std::int64_t replica_count = 0; replica_count < num_replicas; ++replica_count) {
            const auto &vec = system_list_pointer[replica_count]->GetSample();
            for (std::int64_t k = 0; k < system_size; ++k) {
               const std::int64_t ind = replica_count*num_samples*system_size + sample_count*system_size + k;
               samples(ind) = vec[k].GetValue();
            }
         }
      }
      return samples;
   }
};

//! @brief Make ClassicalMonteCarlo class.
//! @tparam ModelType Model class.
//! @return The ClassicalMonteCarlo class.
template<class ModelType>
auto make_classical_monte_carlo() {
   return ClassicalMonteCarlo<ModelType>();
}

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
