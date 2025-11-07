/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 * Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                      *
 *                                                                                               *
 * The IPsolver project is distributed under the MIT License.                                    *
 *                                                                                               *
 * Davide Stocco                                                               Enrico Bertolazzi *
 * University of Trento                                                     University of Trento *
 * e-mail: davide.stocco@unitn.it                             e-mail: enrico.bertolazzi@unitn.it *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// This little script demonstrates the use the primal-dual interior-point solver to compute a logistic
// regression model for predicting the binary {0,1} outputs of a input vectors. It computes the set
// of parameters that maximizes the likelihood (or minimizes the sum of squared errors), but subject
// to a penalization on (otherwise known as the "Lasso" or "Basis pursuit denoising"). The best way
// to understand what is going on here is to go and read:
//
// * Trevor Hastie, Robert Tibshirani and Jerome Friedman (2001). The Elements of Statistical Learning.
//   Springer.
//
// * Scott S. Chen, David L. Donoho and Michael A. Saunders (2001). Atomic Decomposition by Basis
//   Pursuit. SIAM Review, Vol. 43, No. 1. (2001), pp. 129-159.
//
// The computed solution should be fairly close to the "true" regression coefficients (beta). Note
// that the Hessian in this approach is intensely ill-conditioned (due to transformation of the
// regression coefficients into their positive and negative components), so in general this may not
// be the best approach for logistic regression with L1 regularization. The steepest descent direction
// actually works well descent despite a Hessian with a very large condition number. There is probably
// a good reason why, but  at this point I don't know. (Sorry.)
//
//                                                                    Peter Carbonetto
//                                                                    Dept. of Computer Science
//                                                                    University of British Columbia
//                                                                    Copyright 2008

// STL includes
#include <memory>

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
  MatrixH m_Q;
  VectorN m_c;
  MatrixJ m_A;
  VectorM m_b;

public:
  // Constructor
  QuadraticProgram(MatrixH const & Q, VectorN const & c, MatrixJ const & A, VectorM const & b)
    : m_Q(Q), m_c(c), m_A(A), m_b(b) {}

  // Objective function
  bool objective(VectorN const & x, Real & out) const override
  {
    out = 0.5 * x.dot(m_Q * x) + m_c.dot(x);
    return std::isfinite(out);
  }

  // Gradient of the objective function
  bool objective_gradient(VectorN const & x, VectorN & out) const override
  {
    out = m_Q * x + m_c;
    return out.allFinite();
  }

  // Hessian of the objective function
  bool objective_hessian(VectorN const & /*x*/, MatrixH & out) const override
  {
    out = m_Q;
    return out.allFinite();
  }

  // Constraints function
  bool constraints(VectorN const & x, VectorM & out) const override
  {
    out = m_A * x - m_b;
    return out.allFinite();
  }

  // Jacobian of the constraints function
  bool constraints_jacobian(VectorN const & /*x*/, MatrixJ & out) const override
  {
    out = m_A;
    return out.allFinite();
  }

  // Hessian of the Lagrangian function
  bool lagrangian_hessian(VectorN const & /*x*/, VectorM const & /*z*/, MatrixH & out) const override
  {
    out = m_Q;
    return out.allFinite();
  }
};

// Test fixture for IPsolver with QuadraticProgram
class LinearRegression : public testing::Test {
protected:
  using TestType = double;

  static constexpr Integer N = 2;
  static constexpr Integer M = 5;
  static constexpr Integer NType = Eigen::Dynamic; // 2
  static constexpr Integer MType = Eigen::Dynamic; // 5

  using VectorN = typename IPsolver::Problem<TestType, NType, MType>::VectorN;
  using VectorM = typename IPsolver::Problem<TestType, NType, MType>::VectorM;
  using MatrixJ = typename IPsolver::Problem<TestType, NType, MType>::MatrixJ;
  using MatrixH = typename IPsolver::Problem<TestType, NType, MType>::MatrixH;

  std::unique_ptr<QuadraticProgram<TestType, NType, MType>> problem;
  VectorN x_guess, sol;

  void SetUp() override {
    MatrixH Q(2.0 * MatrixH::Identity(N, N));
    VectorN c(N); c << -2.0, -5.0;
    MatrixJ A(M, N);
    A <<  1.0,  2.0,
       -1.0,  2.0,
       -1.0, -2.0,
        1.0,  0.0,
        0.0,  1.0;
    VectorM b(M); b << 6.0, 2.0, 2.0, 3.0, 2.0;

    x_guess.resize(N); x_guess << 0.5, 0.5;
    sol.resize(N); sol << 1.4, 1.7;

    problem = std::make_unique<QuadraticProgram<TestType, NType, MType>>(Q, c, A, b);
  }
};

TEST_F(LinearRegression, DISABLED_Problem_BFGS) {
  IPsolver::Solver<TestType, NType, MType> solver(std::move(problem));
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::BFGS);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(LinearRegression, DISABLED_Problem_Newton) {
  IPsolver::Solver<TestType, NType, MType> solver(std::move(problem));
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::NEWTON);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(LinearRegression, Problem_Steepest) {
  IPsolver::Solver<TestType, NType, MType> solver(std::move(problem));
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, NType, MType>::Descent::STEEPEST);

  VectorN x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
  EXPECT_TRUE(x_sol.isApprox(sol, APPROX_TOLERANCE));
}

TEST_F(LinearRegression, DISABLED_ProblemWrapper_BFGS) {
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

TEST_F(LinearRegression, DISABLED_ProblemWrapper_Newton) {
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

TEST_F(LinearRegression, ProblemWrapper_Steepest) {
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
