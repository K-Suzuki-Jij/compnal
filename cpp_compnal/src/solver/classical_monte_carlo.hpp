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
//  Created by Kohei Suzuki on 2022/06/11.
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_

#include "../utility/type.hpp"
#include "./cmc_utility/all.hpp"
#include <vector>
#include <random>
#include <sstream>

namespace compnal {
namespace solver {

template<class ModelType>
class ClassicalMonteCarlo {
   
public:

   using RealType  = typename ModelType::ValueType;
   using IndexType = typename ModelType::IndexType;
   using OPType    = typename ModelType::OPType;
   
   ClassicalMonteCarlo(const ModelType &model): model_(model) {}
   
   void SetNumSweeps(const std::int32_t num_sweeps) {
      if (num_sweeps < 0) {
         throw std::runtime_error("num_sweeps must be non-negative integer.");
      }
      num_sweeps_ = num_sweeps;
   }
   
   void SetNumSamples(const std::int32_t num_samples) {
      if (num_samples < 0) {
         throw std::runtime_error("num_samples must be non-negative integer.");
      }
      num_samples_ = num_samples;
   }
   
   void SetNumThreads(const std::int32_t num_threads) {
      if (num_threads < 0) {
         throw std::runtime_error("num_threads must be non-negative integer.");
      }
      num_threads_ = num_threads;
   }
   
   void SetTemperature(const RealType temperature) {
      if (temperature <= 0) {
         throw std::runtime_error("Temperature must be positive number");
      }
      beta_ = 1/temperature;
   }
   
   void SetAlgorithm(const cmc_utility::Algorithm algorithm) {
      algorithm_ = algorithm;
   }
   
   std::int32_t GetNumSweeps() const {
      return num_sweeps_;
   }
   
   std::int32_t GetNumSamples() const {
      return num_samples_;
   }
   
   std::int32_t GetNumThreads() const {
      return num_threads_;
   }
   
   RealType GetTemperature() const {
      return 1/beta_;
   }
   
   cmc_utility::Algorithm GetAlgorithm() const {
      return algorithm_;
   }

   std::uint64_t GetSeed() const {
      return seed_;
   }
   
   const std::vector<std::vector<OPType>> &GetSamples() const {
      return samples_;
   }
   
   void Run() {
      Run(std::random_device()());
   }
   
   void Run(const std::uint64_t seed) {
      if (num_samples_ < 0) {
         throw std::runtime_error("num_samples must be non-negative integer.");
      }
      if (num_sweeps_ < 0) {
         throw std::runtime_error("num_sweeps must be non-negative integer.");
      }
      
      seed_ = seed;
      
      utility::RandType random_number_engine(seed_);
      std::vector<std::uint64_t> system_seed_list(num_samples_);
      std::vector<std::uint64_t> mc_seed_list(num_samples_);

      for (int32_t i = 0; i < num_samples_; ++i) {
         system_seed_list[i] = random_number_engine();
         mc_seed_list[i] = random_number_engine();
      }
      
      samples_.clear();
      samples_.shrink_to_fit();
      samples_.resize(num_samples_);
      
      if (algorithm_ == cmc_utility::Algorithm::METROPOLIS) {
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
         for (std::int32_t sample_count = 0; sample_count < num_samples_; sample_count++) {
            cmc_utility::CMCSystem system{model_};
            system.InitializeSSF(system_seed_list[sample_count]);
            cmc_utility::SSFUpdater(&system, num_sweeps_, beta_, mc_seed_list[sample_count], cmc_utility::metropolis_transition<RealType>);
            samples_[sample_count] = system.GetSample();
         }
      }
      else if (algorithm_ == cmc_utility::Algorithm::HEAT_BATH) {
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
         for (std::int32_t sample_count = 0; sample_count < num_samples_; sample_count++) {
            cmc_utility::CMCSystem system{model_};
            system.InitializeSSF(system_seed_list[sample_count]);
            cmc_utility::SSFUpdater(&system, num_sweeps_, beta_, mc_seed_list[sample_count], cmc_utility::heat_bath_transition<RealType>);
            samples_[sample_count] = system.GetSample();
         }
      }
      else if (algorithm_ == cmc_utility::Algorithm::IRKMR) {
         throw std::runtime_error("Under Construction");
      }
      else if (algorithm_ == cmc_utility::Algorithm::RKMR) {
         throw std::runtime_error("Under Construction");
      }
      else if (algorithm_ == cmc_utility::Algorithm::SWENDSEN_WANG) {
         throw std::runtime_error("Under Construction");
      }
      else if (algorithm_ == cmc_utility::Algorithm::WOLFF) {
         throw std::runtime_error("Under Construction");
      }
      else {
         throw std::runtime_error("Unknown Algorithm");
      }

   }
   
   RealType CalculateAverage() const {
      return model_.CalculateMoment(samples_, 1, num_threads_);
   }

   RealType CalculateMoment(const std::int32_t degree) const {
      return model_.CalculateMoment(samples_, degree, num_threads_);
   }
   
   std::vector<RealType> CalculateOnsiteAverage() const {
      const std::int32_t system_size = model_.GetSystemSize();
      std::vector<RealType> dist(system_size);
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int32_t i = 0; i < system_size; ++i) {
         RealType value = 0.0;
         for (std::size_t j = 0; j < samples_.size(); ++j) {
            value += samples_[j][i];
         }
         dist[i] = value/samples_.size();
      }
      return dist;
   }
   
   std::vector<RealType> CalculateCorrelation(const IndexType origin,
                                              const std::vector<IndexType> &index_list) const {
      std::int32_t size = static_cast<std::int32_t>(index_list.size());
      std::vector<RealType> value_list(size);
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int32_t i = 0; i < size; ++i) {
         value_list[i] = model_.CalculateCorrelation(samples_, origin, index_list[i]);
      }
      return value_list;
   }
   
private:
   std::int32_t num_sweeps_  = 1000;
   std::int32_t num_samples_ = 1;
   std::int32_t num_threads_ = 1;
   RealType beta_            = 1;
   cmc_utility::Algorithm algorithm_ = cmc_utility::Algorithm::METROPOLIS;
   std::uint64_t seed_ = std::random_device()();
   const ModelType model_;
   std::vector<std::vector<OPType>> samples_;
      
};


template<class ModelType>
auto make_classical_monte_carlo(const ModelType &model) {
   return ClassicalMonteCarlo<ModelType>{model};
}


} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
