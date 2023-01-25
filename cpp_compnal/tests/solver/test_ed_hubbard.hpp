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
//  test_ed_hubbard.hpp
//  compnal
//
//  Created by kohei on 2023/01/02.
//  
//

#ifndef COMPNAL_TEST_SOLVER_EXACT_DIAGONALIATION_HUBBARD_HPP_
#define COMPNAL_TEST_SOLVER_EXACT_DIAGONALIATION_HUBBARD_HPP_

#include "../../src/solver/exact_diagonalization.hpp"
#include "../../src/model/quantum/hubbard.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(SolverExactDiagonalization, HubbardChain) {
   lattice::Chain chain{10};
   model::quantum::Hubbard<lattice::Chain, double> hubbard{chain};
   hubbard.SetTotalSz(0);
   hubbard.SetTotalElectron(10);
   solver::ExactDiag ed{hubbard};
   ed.SetNumThreads(8);
   
   ed.CalculateGroundState();

   std::cout << ed.CalculateExpectationValue(hubbard.GetOnsiteOperatorNC(), 0) << std::endl;
   std::cout << ed.CalculateCorrelationFunction(hubbard.GetOnsiteOperatorNC(), 0,
                                                hubbard.GetOnsiteOperatorNC(), 1) << std::endl;
}

TEST(SolverExactDiagonalization, HubbardSquare) {
   lattice::Square square{2, 4};
   model::quantum::Hubbard<lattice::Square, double> hubbard{square};
   hubbard.SetTotalSz(0);
   hubbard.SetTotalElectron(4);
   solver::ExactDiag ed{hubbard};
   ed.SetNumThreads(8);
   
   ed.CalculateGroundState();

   std::cout << ed.CalculateExpectationValue(hubbard.GetOnsiteOperatorNC(), {0, 0}) << std::endl;
   std::cout << ed.CalculateCorrelationFunction(hubbard.GetOnsiteOperatorNC(), {0, 0},
                                                hubbard.GetOnsiteOperatorNC(), {0, 1}) << std::endl;
}

TEST(SolverExactDiagonalization, HubbardCubic) {
   lattice::Cubic cubic{2, 2, 2};
   model::quantum::Hubbard<lattice::Cubic, double> hubbard{cubic};
   hubbard.SetTotalSz(0);
   hubbard.SetTotalElectron(6);
   solver::ExactDiag ed{hubbard};
   ed.SetNumThreads(8);
   
   ed.CalculateGroundState();

   std::cout << ed.CalculateExpectationValue(hubbard.GetOnsiteOperatorNC(), {0, 0, 0}) << std::endl;
   std::cout << ed.CalculateCorrelationFunction(hubbard.GetOnsiteOperatorNC(), {0, 0, 0},
                                                hubbard.GetOnsiteOperatorNC(), {0, 1, 0}) << std::endl;
}

TEST(SolverExactDiagonalization, HeisenbergChain) {
   lattice::Chain chain{4};
   model::quantum::Heisenberg<lattice::Chain, double> heisenberg{chain};
   heisenberg.SetMagnitudeSpin(1.5);
   heisenberg.SetTotalSz(0);
   solver::ExactDiag ed{heisenberg};
   ed.SetNumThreads(8);
   
   ed.CalculateGroundState();

   std::cout << ed.CalculateExpectationValue(heisenberg.GetOnsiteOperatorSz(), 0) << std::endl;
   std::cout << ed.CalculateCorrelationFunction(heisenberg.GetOnsiteOperatorSx(), 0,
                                                heisenberg.GetOnsiteOperatorSx(), 1) << std::endl;
}

TEST(SolverExactDiagonalization, HeisenbergSquare) {
   lattice::Square square{4, 4};
   model::quantum::Heisenberg<lattice::Square, double> heisenberg{square};
   heisenberg.SetTotalSz(0);
   solver::ExactDiag ed{heisenberg};
   ed.SetNumThreads(8);
   
   ed.CalculateGroundState();

   std::cout << ed.CalculateExpectationValue(heisenberg.GetOnsiteOperatorSz(), {0, 0}) << std::endl;
   std::cout << ed.CalculateCorrelationFunction(heisenberg.GetOnsiteOperatorSx(), {0, 0},
                                                heisenberg.GetOnsiteOperatorSx(), {0, 1}) << std::endl;
}

TEST(SolverExactDiagonalization, HeisenbergCubic) {
   lattice::Cubic cubic{2, 3, 2};
   model::quantum::Heisenberg<lattice::Cubic, double> heisenberg{cubic};
   heisenberg.SetTotalSz(0);
   solver::ExactDiag ed{heisenberg};
   ed.SetNumThreads(8);
   
   ed.CalculateGroundState();

   std::cout << ed.CalculateExpectationValue(heisenberg.GetOnsiteOperatorSz(), {0, 0, 0}) << std::endl;
   std::cout << ed.CalculateCorrelationFunction(heisenberg.GetOnsiteOperatorSx(), {0, 0, 0},
                                                heisenberg.GetOnsiteOperatorSx(), {0, 1, 0}) << std::endl;
}


} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_SOLVER_EXACT_DIAGONALIATION_HUBBARD_HPP_ */
