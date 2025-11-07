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
#include <random>

// GTest library
#include <gtest/gtest.h>

// IPsolver includes
#include "IPsolver.hh"

using IPsolver::Integer;

constexpr bool VERBOSE{true};
constexpr double SOLVER_TOLERANCE{1.0e-6};
constexpr Integer MAX_ITERATIONS{100};

// Logistic regression class
template<typename Real, int N, int M, int D>
class LogisticRegression : public IPsolver::Problem<Real, N, M>
{
public:
  using typename IPsolver::Problem<Real, N, M>::VectorN;
  using typename IPsolver::Problem<Real, N, M>::VectorM;
  using typename IPsolver::Problem<Real, N, M>::MatrixH;
  using typename IPsolver::Problem<Real, N, M>::MatrixJ;

  using VectorD = Eigen::Vector<Real, D>;
  using MatrixP = Eigen::Matrix<Real, D, M>;

private:
  MatrixP m_P; // Input data matrix
  VectorD m_y; // Binary response vector
  Real m_lambda; // Regularization parameter

public:
  // Constructor
  LogisticRegression(MatrixP const & P, VectorD const & y, Real const lambda)
    : m_P(P), m_y(y), m_lambda(lambda) {}

  // Logistic function
  VectorD logit(VectorD const & x) const {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
  }

  // Objective function: negative log-likelihood + L2 regularization
  bool objective(VectorM const & x, Real & out) const override
  {
    VectorD u(this->logit(m_P * x));
    out = -(this->m_y.array() * u.array().log() + (1.0 - this->m_y.array()) * (1.0 - u.array()).log()).sum()
      + this->m_lambda * x.sum();
    return std::isfinite(out);
  }

  // Gradient of the objective function
  bool objective_gradient(VectorM const & x, VectorM & out) const override
  {
    VectorD u(this->logit(m_P * x));
    out = ((-this->m_P.transpose() * (this->m_y - u)).array() + this->m_lambda).matrix();
    return out.allFinite();
  }

  // Hessian of the objective function
  bool objective_hessian(VectorM const & x, MatrixH & out) const override
  {
    VectorD u(this->logit(m_P * x));
    Eigen::DiagonalMatrix<Real, D> U((u.array() * (1.0 - u.array())).matrix().asDiagonal());
    out = this->m_P.transpose() * U * this->m_P;
    return out.allFinite();
  }

  // Constraints function
  bool constraints(VectorM const & x, VectorM & out) const override
  {
    out = -x;
    return out.allFinite();
  }

  // Jacobian of the constraints function
  bool constraints_jacobian(VectorM const & x, MatrixJ & out) const override
  {
    out = -MatrixJ::Identity(x.size(), x.size());
    return out.allFinite();
  }

  // Hessian of the Lagrangian function
  bool lagrangian_hessian(VectorM const & x, VectorM const & /*z*/, MatrixH & out) const override
  {
    out = MatrixH::Zero(x.size(), x.size());
    return out.allFinite();
  }
};

// Logistic regression example
class LogisticRegressionTest : public testing::Test {
protected:
  using TestType = double;

  static constexpr Integer N = 8;
  static constexpr Integer M = 16;
  static constexpr Integer D = 100;
  static constexpr Integer NType = Eigen::Dynamic; // 8
  static constexpr Integer MType = Eigen::Dynamic; // 16
  static constexpr Integer DType = Eigen::Dynamic; // 100

  using VectorM = typename IPsolver::Problem<TestType, MType, MType>::VectorM;
  using MatrixJ = typename IPsolver::Problem<TestType, MType, MType>::MatrixJ;
  using MatrixH = typename IPsolver::Problem<TestType, MType, MType>::MatrixH;
  using VectorN = Eigen::Vector<TestType, N>;
  using VectorD = Eigen::Vector<TestType, D>;
  using MatrixA = Eigen::Matrix<TestType, D, N>;
  using MatrixP = Eigen::Matrix<TestType, D, M>;

  std::unique_ptr<LogisticRegression<TestType, MType, MType, DType>> problem;
  VectorM x_guess;

  void SetUp() override
  {
    constexpr TestType epsilon{0.25};

    VectorN beta(N);
    beta << 0.0, 0.0, 2.0, -4.0, 0.0, 0.0, -1.0, 3.0;

    VectorN sigma(N);
    sigma << 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    std::mt19937 gen(42);
    std::normal_distribution<TestType> normal(0.0, 1.0);
    MatrixA A(D, N);
    for (Integer i{0}; i < D; ++i) {
      for (Integer j{0}; j < N; ++j) {
        A(i, j) = sigma(j) * normal(gen);
      }
    }

    VectorD noise(D);
    for (Integer i{0}; i < D; ++i) {
      noise(i) = epsilon * normal(gen);
    }

    auto logit = [](VectorD const & x) -> VectorD {
      return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    };

    VectorD y(logit(A * beta + noise));
    for (Integer i{0}; i < D; ++i) {
      y(i) = (normal(gen) < y(i)) ? 1.0 : 0.0;
    }

    constexpr TestType lambda{0.5};
    MatrixP P(D, M);
    P << A, -A;

    x_guess = VectorM::Ones(M);

    problem = std::make_unique<LogisticRegression<TestType, MType, MType, DType>>(P, y, lambda);
  }
};

TEST_F(LogisticRegressionTest, ProblemClass) {
  IPsolver::Solver<TestType, MType, MType> solver(std::move(problem));
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, MType, MType>::Descent::NEWTON);

  VectorM x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
}

TEST_F(LogisticRegressionTest, ProblemWrapperClass) {
  IPsolver::ProblemWrapper<TestType, MType, MType> problem_wrapper(
    [this] (VectorM const & x, TestType & out) {return this->problem->objective(x, out);},
    [this] (VectorM const & x, VectorM & out) {return this->problem->objective_gradient(x, out);},
    [this] (VectorM const & x, MatrixH & out) {return this->problem->objective_hessian(x, out);},
    [this] (VectorM const & x, VectorM & out) {return this->problem->constraints(x, out);},
    [this] (VectorM const & x, MatrixJ & out) {return this->problem->constraints_jacobian(x, out);},
    [this] (VectorM const & x, VectorM const & z, MatrixH & out) {return this->problem->lagrangian_hessian(x, z, out);}
  );

  IPsolver::Solver<TestType, MType, MType> solver(
    problem_wrapper.objective(), problem_wrapper.objective_gradient(), problem_wrapper.objective_hessian(),
    problem_wrapper.constraints(), problem_wrapper.constraints_jacobian(), problem_wrapper.lagrangian_hessian()
  );
  solver.verbose_mode(VERBOSE);
  solver.tolerance(SOLVER_TOLERANCE);
  solver.max_iterations(MAX_ITERATIONS);
  solver.descent(IPsolver::Solver<TestType, MType, MType>::Descent::NEWTON);

  VectorM x_sol;
  EXPECT_TRUE(solver.solve(x_guess, x_sol));
}
