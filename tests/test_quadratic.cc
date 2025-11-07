/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 * Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                      *
 *                                                                                               *
 * The IPsolver project is distributed under the MIT License.                                    *
 *                                                                                               *
 * Davide Stocco                                                               Enrico Bertolazzi *
 * University of Trento                                                     University of Trento *
 * e-mail: davide.stocco@unitn.it                             e-mail: enrico.bertolazzi@unitn.it *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// This short script demonstrates the use of the interior-point solver to compute the solution to a
// quadratic program with convex objective (i.e. positive-definite Hessian) and convex, quadratic
// inequality constraints. More precisely, it finds the solution to the following
// optimization problem:
//
//   minimize    (1/2)x'Hx + q'x
//   subject to  ci(x) < b
//
// where the inequality constraints are quadratic functions
//
//   ci(x) = (1/2)x'P{i}x + r{i}'x
//
// and the quantities {H,q,P,r,b} are all specified below. Note that this code is not a particular
// efficient way to optimize a constrained quadratic program, and should not be used for solving
// large optimization problems. This particular example originally comes from the book:
//
//   H. P. Schwefel (1995) Evolution and Optimum Seeking.
//
// The minimium occurs at (0,1,2,-1).
//
//                                                                    Peter Carbonetto
//                                                                    Dept. of Computer Science
//                                                                    University of British Columbia
//                                                                    Copyright 2008

// STL includes
#include <vector>

// GTest library
#include <gtest/gtest.h>

// IPsolver includes
#include "IPsolver.hh"

using IPsolver::Integer;

constexpr bool VERBOSE{true};
constexpr double SOLVER_TOLERANCE{5.0e-5};
constexpr double APPROX_TOLERANCE{1.0e-4};
constexpr Integer MAX_ITERATIONS{100};

// Quadratic program class
template<typename Real, Integer N, Integer M>
class QuadraticProgram : public IPsolver::Problem<Real, N, M>
{
public:
  using typename IPsolver::Problem<Real, N, M>::VectorN;
  using typename IPsolver::Problem<Real, N, M>::VectorM;
  using typename IPsolver::Problem<Real, N, M>::MatrixH;
  using typename IPsolver::Problem<Real, N, M>::MatrixJ;

private:
  MatrixH m_H;              // Hessian of the objective function
  VectorN m_q;              // Gradient of the objective function
  std::vector<MatrixH> m_P; // Hessian matrices for the constraints
  std::vector<VectorN> m_r; // Gradient vectors for the constraints
  VectorM m_b;              // Right-hand side vector for the constraints

public:
  // Constructor
  QuadraticProgram(MatrixH const & H, VectorN const & q, std::vector<MatrixH> const & P,
    std::vector<VectorN> const & r, VectorM const & b)
    : m_H(H), m_q(q), m_P(P), m_r(r), m_b(b) {}

  // Objective function
  bool objective(VectorN const & x, Real & out) const override
  {
    out = 0.5 * x.dot(this->m_H * x) + this->m_q.dot(x);
    return std::isfinite(out);
  }

  // Gradient of the objective function
  bool objective_gradient(VectorN const & x, VectorN & out) const override
  {
    out = this->m_H * x + this->m_q;
    return out.allFinite();
  }

  // Hessian of the objective function
  bool objective_hessian(VectorN const & /*x*/, MatrixH & out) const override
  {
    out = this->m_H;
    return out.allFinite();
  }

  // Constraints function
  bool constraints(VectorN const & x, VectorM & out) const override
  {
    out = VectorM::Zero(static_cast<Integer>(this->m_b.size()));
    for (Integer i{0}; i < static_cast<Integer>(this->m_b.size()); ++i) {
      out(i) = 0.5 * x.dot(this->m_P[i] * x) + this->m_r[i].dot(x) - this->m_b(i);
    }
    return out.allFinite();
  }

  // Jacobian of the constraints function
  bool constraints_jacobian(VectorN const & x, MatrixJ & out) const override
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(this->m_b.size())};
    out = MatrixJ::Zero(m, n);
    for (Integer i{0}; i < m; ++i) {
      out.row(i) = (this->m_P[i] * x + this->m_r[i]).transpose();
    }
    return out.allFinite();
  }

  // Hessian of the Lagrangian function
  bool lagrangian_hessian(VectorN const & x, VectorM const & z, MatrixH & out) const override
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(this->m_b.size())};
    out = MatrixH::Zero(n, n);
    for (Integer i{0}; i < m; ++i) {
      out += z(i) * this->m_P[i];
    }
    return out.allFinite();
  }

}; // QuadraticProgram class

// Define the quadratic program example
class QuadraticProgramTest : public testing::Test {
protected:
  using TestType = double;

  static constexpr Integer N = 4;
  static constexpr Integer M = 3;
  static constexpr Integer NType =  Eigen::Dynamic; // 4
  static constexpr Integer MType =  Eigen::Dynamic; // 3

  using VectorN = typename IPsolver::Problem<TestType, NType, MType>::VectorN;
  using VectorM = typename IPsolver::Problem<TestType, NType, MType>::VectorM;
  using MatrixJ = typename IPsolver::Problem<TestType, NType, MType>::MatrixJ;
  using MatrixH = typename IPsolver::Problem<TestType, NType, MType>::MatrixH;

  std::unique_ptr<QuadraticProgram<TestType, NType, MType>> problem;
  VectorN x_guess;
  VectorN sol;

  void SetUp() override {

  MatrixH H(MatrixH::Zero(N, N));
  H.diagonal() << 2.0, 2.0, 4.0, 2.0;
  VectorN q(N);
    q << -5.0, -5.0, -21.0, 7.0;

  std::vector<MatrixH> P(M, MatrixH::Zero(N, N));
    P[0].diagonal() << 4.0, 2.0, 2.0, 0.0;
    P[1].diagonal() << 2.0, 2.0, 2.0, 2.0;
    P[2].diagonal() << 2.0, 4.0, 2.0, 4.0;

  std::vector<VectorN> r(M, VectorN::Zero(N));
    r[0] <<  2.0, -1.0,  0.0, -1.0;
    r[1] <<  1.0, -1.0,  1.0, -1.0;
    r[2] << -1.0,  0.0,  0.0, -1.0;

  VectorM b(M);
    b << 5.0, 8.0, 10.0;

    x_guess = VectorN::Zero(N);
    sol.resize(N);
    sol << 0.0, 1.0, 2.0, -1.0;

    problem = std::make_unique<QuadraticProgram<TestType, NType, MType>>(H, q, P, r, b);
  }
};

TEST_F(QuadraticProgramTest, Problem_BFGS) {
  IPsolver::Solver<TestType, NType, MType> solver(std::move(problem));
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::BFGS);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(QuadraticProgramTest, Problem_Newton) {
  IPsolver::Solver<TestType, NType, MType> solver(std::move(problem));
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::NEWTON);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(QuadraticProgramTest, Problem_Steepest) {
  IPsolver::Solver<TestType, NType, MType> solver(std::move(problem));
  solver.tolerance(1e-6);
  solver.max_iterations(100);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::STEEPEST);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(QuadraticProgramTest, Wrapper_BFGS) {
  IPsolver::ProblemWrapper<TestType, NType, MType> problem_wrapper(
    [this] (const VectorN & x, TestType & out) {return this->problem->objective(x, out);},
    [this] (const VectorN & x, VectorN & out) {return this->problem->objective_gradient(x, out);},
    [this] (const VectorN & x, MatrixH & out) {return this->problem->objective_hessian(x, out);},
    [this] (const VectorN & x, VectorM & out) {return this->problem->constraints(x, out);},
    [this] (const VectorN & x, MatrixJ & out) {return this->problem->constraints_jacobian(x, out);},
    [this] (const VectorN & x, const VectorM & z, MatrixH & out) {return this->problem->lagrangian_hessian(x, z, out);}
  );

  IPsolver::Solver<TestType, NType, MType> solver(
    problem_wrapper.objective(), problem_wrapper.objective_gradient(), problem_wrapper.objective_hessian(),
    problem_wrapper.constraints(), problem_wrapper.constraints_jacobian(), problem_wrapper.lagrangian_hessian()
  );
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::BFGS);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(QuadraticProgramTest, Wrapper_Newton) {
  IPsolver::ProblemWrapper<TestType, NType, MType> problem_wrapper(
    [this] (const VectorN & x, TestType & out) {return this->problem->objective(x, out);},
    [this] (const VectorN & x, VectorN & out) {return this->problem->objective_gradient(x, out);},
    [this] (const VectorN & x, MatrixH & out) {return this->problem->objective_hessian(x, out);},
    [this] (const VectorN & x, VectorM & out) {return this->problem->constraints(x, out);},
    [this] (const VectorN & x, MatrixJ & out) {return this->problem->constraints_jacobian(x, out);},
    [this] (const VectorN & x, const VectorM & z, MatrixH & out) {return this->problem->lagrangian_hessian(x, z, out);}
  );

  IPsolver::Solver<TestType, NType, MType> solver(
    problem_wrapper.objective(), problem_wrapper.objective_gradient(), problem_wrapper.objective_hessian(),
    problem_wrapper.constraints(), problem_wrapper.constraints_jacobian(), problem_wrapper.lagrangian_hessian()
  );
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::NEWTON);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(QuadraticProgramTest, Wrapper_Steepest) {
  IPsolver::ProblemWrapper<TestType, NType, MType> problem_wrapper(
    [this] (const VectorN & x, TestType & out) {return this->problem->objective(x, out);},
    [this] (const VectorN & x, VectorN & out) {return this->problem->objective_gradient(x, out);},
    [this] (const VectorN & x, MatrixH & out) {return this->problem->objective_hessian(x, out);},
    [this] (const VectorN & x, VectorM & out) {return this->problem->constraints(x, out);},
    [this] (const VectorN & x, MatrixJ & out) {return this->problem->constraints_jacobian(x, out);},
    [this] (const VectorN & x, const VectorM & z, MatrixH & out) {return this->problem->lagrangian_hessian(x, z, out);}
  );

  IPsolver::Solver<TestType, NType, MType> solver(
    problem_wrapper.objective(), problem_wrapper.objective_gradient(), problem_wrapper.objective_hessian(),
    problem_wrapper.constraints(), problem_wrapper.constraints_jacobian(), problem_wrapper.lagrangian_hessian()
  );
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::STEEPEST);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}
