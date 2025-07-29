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

// Catch2 library
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/catch_template_test_macros.hpp>

// IPsolver includes
#include "IPsolver.hh"

using IPsolver::Integer;

// Logistic regression class
template<typename T>
class LogisticRegression : public IPsolver::Problem<T>
{
public:
  using typename IPsolver::Problem<T>::Vector;
  using typename IPsolver::Problem<T>::Matrix;

private:
  Matrix m_P; // Input data matrix
  Vector m_y; // Binary response vector
  T m_lambda; // Regularization parameter

public:
  // Constructor
  LogisticRegression(const Matrix& P, const Vector& y, T lambda)
    : m_P(P), m_y(y), m_lambda(lambda) {}

  // Logistic function
  Vector logit(const Vector& x) const {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
  }

  // Objective function
  T objective(const Vector& x) const override
  {
    Vector u(logit(this->m_P * x));
    return -(this->m_y.array() * u.array().log() + (1.0 - this->m_y.array()) * (1.0 - u.array()).log()).sum()
      + this->m_lambda * x.sum();
  }

  // Gradient of the objective function
  Vector objective_gradient(const Vector& x) const override
  {
    Vector u(logit(this->m_P * x));
    return ((-this->m_P.transpose() * (this->m_y - u)).array() + this->m_lambda).matrix();
  }

  // Hessian of the objective function
  Matrix objective_hessian(const Vector& x) const override
  {
    Vector u(logit(this->m_P * x));
    Matrix U((u.array() * (1.0 - u.array())).matrix().asDiagonal());
    return this->m_P.transpose() * U * this->m_P;
  }

  // Constraints function
  Vector constraints(const Vector& x) const override
  {
    return -x;
  }

  // Jacobian of the constraints function
  Matrix constraints_jacobian(const Vector& x, const Vector& /*z*/) const override
  {
    return -Matrix::Identity(x.size(), x.size());
  }

  // Hessian of the Lagrangian function
  Matrix lagrangian_hessian(const Vector& x, const Vector& /*z*/) const override
  {
    return Matrix::Zero(x.size(), x.size());
  }

}; // LogisticRegression class

// Logistic regression example
TEMPLATE_TEST_CASE("IPsolver logistic regression example", "[template]", double) { // float, double, long double
  using Integer = typename IPsolver::Integer;
  using Vector  = typename IPsolver::Problem<TestType>::Vector;
  using Matrix  = typename IPsolver::Problem<TestType>::Matrix;

  // Define the logistic function
  auto logit = [](const Vector& x) -> Vector {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
  };

  // CREATE DATA SET
  // Generate the input vectors from the standard normal, and generate the binary responses from the
  // regression with some additional noise, and then transform the results using the logistic function.
  // The variable "beta" is the set of true regression coefficients of length m.
  constexpr Integer m{8};
  constexpr Integer n{100};
  constexpr TestType epsilon{0.25};

  // True regression coefficients
  Vector beta(m);
  beta << 0.0, 0.0, 2.0, -4.0, 0.0, 0.0, -1.0, 3.0;

  // Standard deviation of coordinates
  Vector sigma(m);
  sigma << 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  // The n x m matrix of samples
  std::mt19937 gen(42);
  std::normal_distribution<TestType> normal(0.0, 1.0);
  Matrix A(n, m);
  for (Integer i{0}; i < n; ++i) {
    for (Integer j{0}; j < m; ++j) {
      A(i, j) = sigma(j) * normal(gen);
    }
  }

  // Noise in outputs
  Vector noise(n);
  for (Integer i{0}; i < n; ++i) {
    noise(i) = epsilon * normal(gen);
  }

  // The binary outputs
  Vector y(logit(A * beta + noise));
  for (Integer i{0}; i < n; ++i) {
    y(i) = (normal(gen) < y(i)) ? 1.0 : 0.0;
  }

  // COMPUTE SOLUTION WITH INTERIOR-POINT METHOD.
  // Compute the L1-regularized maximum likelihood estimator
  constexpr TestType lambda{0.5};
  Matrix P(n, 2*m);
  P << A, -A;

  // Solve the optimization problem
  Vector x_guess(Vector::Ones(2*m));

  // Create the problem object
  std::unique_ptr<LogisticRegression<TestType>> problem{
    std::make_unique<LogisticRegression<TestType>>(P, y, lambda)
  };

  // Create a problem wrapper object
  IPsolver::ProblemWrapper<TestType> problem_wrapper(
    [&problem](const Vector &x) {return problem->objective(x);},
    [&problem](const Vector &x) {return problem->objective_gradient(x);},
    [&problem](const Vector &x) {return problem->objective_hessian(x);},
    [&problem](const Vector &x) {return problem->constraints(x);},
    [&problem](const Vector &x, const Vector &z) {return problem->constraints_jacobian(x, z);},
    [&problem](const Vector &x, const Vector &z) {return problem->lagrangian_hessian(x, z);}
  );

  // Solve the optimization problem with the problem class
  SECTION("Problem Class") {

    // Create the solver
    IPsolver::Solver<TestType> solver(std::move(problem));
    solver.tolerance(1e-6);
    solver.max_iterations(100);

    // Does not work with this example, but kept for consistency
    // SECTION("BFGS Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    // }

    // Does not work with this example, but kept for consistency
    SECTION("Newton Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
    }

    // Does not work with this example, but kept for consistency
    // SECTION("Steepest Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    // }
  }

  // Create the quadratic program with the problem wrapper class
  SECTION("Problem Wrapper Class")
  {
    IPsolver::Solver<TestType> solver(
      problem_wrapper.objective(), problem_wrapper.objective_gradient(), problem_wrapper.objective_hessian(),
      problem_wrapper.constraints(), problem_wrapper.constraints_jacobian(), problem_wrapper.lagrangian_hessian()
    );
    solver.tolerance(1e-6);
    solver.max_iterations(100);

    // SECTION("BFGS Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    // }

    SECTION("Newton Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
    }

    // SECTION("Steepest Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    // }
  }
}
