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
//  Created by Kohei Suzuki on 2022/06/11.
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_

#include "../utility/type.hpp"
#include "./utility_cmc/all.hpp"
#include <vector>
#include <random>
#include <sstream>

namespace compnal {
namespace solver {

//! @brief Class for executing classical monte carlo simulation.
//! @tparam ModelType The type of classical models.
template<class ModelType>
class ClassicalMonteCarlo {
   
   //! @brief The value type.
   using ValueType = typename ModelType::ValueType;
   
   //! @brief Coordinate index type.
   using IndexType = typename ModelType::IndexType;
   
   //! @brief The operator type
   using OPType = typename ModelType::OPType;
   
public:
   
   //! @brief Constructor for ClassicalMonteCarlo class.
   //! @param model The classical model.
   ClassicalMonteCarlo(const ModelType &model): model_(model) {}
   
   //! @brief Set the number of sweeps.
   //! @param num_sweeps The number of sweeps, which must be non-negative integer.
   void SetNumSweeps(const std::int32_t num_sweeps) {
      if (num_sweeps < 0) {
         throw std::runtime_error("num_sweeps must be non-negative integer.");
      }
      num_sweeps_ = num_sweeps;
   }
   
   //! @brief Set the number of samples.
   //! @param num_samples The number of samples, which must be non-negative integer.
   void SetNumSamples(const std::int32_t num_samples) {
      if (num_samples < 0) {
         throw std::runtime_error("num_samples must be non-negative integer.");
      }
      num_samples_ = num_samples;
   }
   
   //! @brief Set the number of threads in the calculation.
   //! @param num_threads The number of threads in the calculation, which must be larger than zero.
   void SetNumThreads(const std::int32_t num_threads) {
      if (num_threads <= 0) {
         throw std::runtime_error("num_threads must be non-negative integer.");
      }
      num_threads_ = num_threads;
   }
   
   //! @brief Set the temperature.
   //! @param temperature The temperature, which must be larger than zero.
   void SetTemperature(const ValueType temperature) {
      if (temperature <= 0) {
         throw std::runtime_error("Temperature must be positive number");
      }
      beta_ = 1/temperature;
   }
   
   //! @brief Set the algorithm used in the state update.
   //! @param algorithm The algorithm.
   void SetAlgorithm(const utility_cmc::Algorithm algorithm) {
      algorithm_ = algorithm;
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
   ValueType GetTemperature() const {
      return 1/beta_;
   }
   
   //! @brief Get the algorithm used in the state update.
   //! @return The algorithm.
   utility_cmc::Algorithm GetAlgorithm() const {
      return algorithm_;
   }
   
   //! @brief Get the seed to be used in the calculation.
   //! @return The seed.
   std::uint64_t GetSeed() const {
      return seed_;
   }
   
   //! @brief Get the samples.
   //! @return The samples.
   const std::vector<std::vector<OPType>> &GetSamples() const {
      return samples_;
   }
   
   //! @brief Execute sampling.
   //! Seed to be used in the calculation will be set automatically.
   void Run() {
      Run(std::random_device()());
   }
   
   //! @brief Execute sampling.
   //! @param seed The seed to be used in the calculation.
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
      
      for (std::int32_t i = 0; i < num_samples_; ++i) {
         system_seed_list[i] = random_number_engine();
         mc_seed_list[i] = random_number_engine();
      }
      
      samples_.clear();
      samples_.shrink_to_fit();
      samples_.resize(num_samples_);
      
      if (algorithm_ == utility_cmc::Algorithm::METROPOLIS) {
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
         for (std::int32_t sample_count = 0; sample_count < num_samples_; sample_count++) {
            utility_cmc::CMCSystem system{model_};
            system.InitializeSSF(system_seed_list[sample_count]);
            utility_cmc::SSFUpdater(&system, num_sweeps_, beta_, mc_seed_list[sample_count],
                                    utility_cmc::metropolis_transition<ValueType>);
            samples_[sample_count] = system.GetSample();
         }
      }
      else if (algorithm_ == utility_cmc::Algorithm::HEAT_BATH) {
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
         for (std::int32_t sample_count = 0; sample_count < num_samples_; sample_count++) {
            utility_cmc::CMCSystem system{model_};
            system.InitializeSSF(system_seed_list[sample_count]);
            utility_cmc::SSFUpdater(&system, num_sweeps_, beta_, mc_seed_list[sample_count],
                                    utility_cmc::heat_bath_transition<ValueType>);
            samples_[sample_count] = system.GetSample();
         }
      }
      else if (algorithm_ == utility_cmc::Algorithm::IRKMR) {
         throw std::runtime_error("Under Construction");
      }
      else if (algorithm_ == utility_cmc::Algorithm::RKMR) {
         throw std::runtime_error("Under Construction");
      }
      else if (algorithm_ == utility_cmc::Algorithm::SWENDSEN_WANG) {
         throw std::runtime_error("Under Construction");
      }
      else if (algorithm_ == utility_cmc::Algorithm::WOLFF) {
         throw std::runtime_error("Under Construction");
      }
      else {
         throw std::runtime_error("Unknown Algorithm");
      }
      
   }
   
   //! @brief Calculate average value for samples.
   //! This function calculates the following value.
   //! \f[ \frac{1}{\rm{num\_samples}}\sum^{\rm{num\_samples}}_{i=1}m_{i}
   //! = \frac{1}{\rm{num\_samples}}\sum^{\rm{num\_samples}}_{i=1}\frac{1}{N}\sum^{N}_{j=1}s^{(i)}_{j}\f]
   //! Here, \f$ s^{(i)}_{j}\f$ is the value of the sample at the \f$j\f$ site for the \f$ i\f$-th sample.
   //! @return Average value for samples.
   ValueType CalculateSampleAverage() const {
      ValueType avg = 0;
#pragma omp parallel for schedule(guided) reduction(+: avg) num_threads(num_threads_)
      for (std::size_t i = 0; i < samples_.size(); ++i) {
         ValueType temp_avg = 0;
         for (std::size_t j = 0; j < samples_[i].size(); ++j) {
            temp_avg += samples_[i][j];
         }
         avg += temp_avg/samples_[i].size();
      }
      return avg/samples_.size();
   }
   
   //! @brief Calculate \f$ n\f$-th order moment for samples.
   //! This function calculates the following value.
   //! \f[ \frac{1}{\rm{num\_samples}}\sum^{\rm{num\_samples}}_{i=1}m^{n}_{i}
   //! = \frac{1}{\rm{num\_samples}}\sum^{\rm{num\_samples}}_{i=1}\left(\frac{1}{N}\sum^{N}_{j=1}s^{(i)}_{j}\right)^n\f]
   //! Here, \f$ s^{(i)}_{j}\f$ is the value of the sample at the \f$j\f$ site for the \f$ i\f$-th sample,
   //! and \f$ n \f$ is the order of moment.
   //! @param order The order of moment \f$ n \f$.
   //! @return Moment for samples.
   ValueType CalculateSampleMoment(const std::int32_t order) const {
      if (order <= 0) {
         throw std::runtime_error("order must be lager than 0.");
      }
      ValueType val = 0;
#pragma omp parallel for schedule(guided) reduction(+: val) num_threads(num_threads_)
      for (std::size_t i = 0; i < samples_.size(); ++i) {
         ValueType avg = 0;
         for (std::size_t j = 0; j < samples_[i].size(); ++j) {
            avg += samples_[i][j];
         }
         avg = avg/samples_[i].size();
         
         ValueType prod = 1;
         for (std::int32_t j = 0; j < order; ++j) {
            prod = prod*avg;
         }
         val += prod;
      }
      return val/samples_.size();
   }
   
   //! @brief Calculate correlations for samples between sites.
   //! This function calculates the following values for the site \f$ r\f$.
   //! \f[ \frac{1}{\rm{num\_samples}}\sum^{\rm{num\_samples}}_{i=1} s^{(i)}_{o}s^{(i)}_{r} \f]
   //! Here, \f$ s^{(i)}_{o}\f$ is the value of the sample at the \f$o\f$ site for the \f$ i\f$-th sample.
   //! @param origin The orign of correlation \f$ o\f$.
   //! @param index_list The list of sites \f$ r\f$.
   //! @return Correlations for samples between sites.
   std::vector<ValueType> CalculateSampleCorrelation(const IndexType origin,
                                                     const std::vector<IndexType> &index_list) const {
      
      std::vector<ValueType> value_list(index_list.size());
      
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::size_t i = 0; i < index_list.size(); ++i) {
         const std::int32_t origin_index = model_.CalculateIntegerSiteIndex(origin);
         const std::int32_t corr_index   = model_.CalculateIntegerSiteIndex(index_list[i]);
         for (std::size_t j = 0; j < samples_.size(); ++j) {
            value_list[i] += samples_[j][origin_index]*samples_[j][corr_index];
         }
         value_list[i] = value_list[i]/samples_.size();
      }
      
      return value_list;
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
   
   //! @brief The inverse temperature.
   ValueType beta_ = 1;
   
   //! @brief The algorithm used in the state update.
   utility_cmc::Algorithm algorithm_ = utility_cmc::Algorithm::METROPOLIS;
   
   //! @brief The seed to be used in the calculation.
   std::uint64_t seed_ = std::random_device()();
   
   //! @brief The samples.
   std::vector<std::vector<OPType>> samples_ = std::vector<std::vector<OPType>>(num_samples_);
   
};


//! @brief Helper function to make ClassicalMonteCarlo class.
//! @tparam ModelType The type of model.
//! @param model The model.
//! @return ClassicalMonteCarlo class.
template<class ModelType>
auto make_classical_monte_carlo(const ModelType &model) {
   return ClassicalMonteCarlo<ModelType>{model};
}


} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
