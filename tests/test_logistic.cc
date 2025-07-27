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

// Logistic regression example
TEMPLATE_TEST_CASE("IPsolver logistic regression example", "[template]", double, float) {
  using Integer = typename IPsolver::Integer;
  using Vector  = typename IPsolver::Solver<TestType>::Vector;
  using Matrix  = typename IPsolver::Solver<TestType>::Matrix;

  // Define the number of training examples and features
  constexpr Integer m{8};
  constexpr Integer n{100};
  constexpr TestType epsilon{0.25};
  constexpr TestType lambda{0.5};

  Vector beta(m);
  beta << 0, 0, 2, -4, 0, 0, -1, 3;

  Vector sigma(m);
  sigma << 10, 1, 1, 1, 1, 1, 1, 1;

  std::mt19937 gen(42);
  std::normal_distribution<TestType> normal(0, 1);

  Matrix A(n, m);
  for (Integer i{0}; i < n; ++i) {
    for (Integer j{0}; j < m; ++j) {
      A(i, j) = sigma(j) * normal(gen);
    }
  }

  Vector noise(n);
  for (Integer i{0}; i < n; ++i) {
    noise(i) = epsilon * normal(gen);
  }

  auto logit = [](const Vector& x) -> Vector {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
  };

  std::cout << "A * beta + noise = " << (logit(A * beta + noise).array() > 0.5) << std::endl;

  Vector y;//(((logit(A * beta + noise).array()) > 0.5).template cast<TestType>());

  Matrix P(n, 2*m);
  P << A, -A;

  // Objective function
  auto objective = [=](const Vector& x) -> TestType { //&P, &y, &lambda, &logit
    Vector u(logit(P * x));
    return -(y.array() * u.array().log() + (1.0 - y.array()) * (1.0 - u.array()).log()).sum()
      + lambda * x.sum();
  };

  // Gradient of the objective
  auto objective_gradient = [=](const Vector& x) -> Vector { //&P, &y, &lambda, &logit
    Vector u(logit(P * x));
    return -P.transpose() * (y - u) + lambda * Vector::Ones(x.size());
  };

  // Constraints ci(x) = -xᵢ < 0
  auto constraints = [](const Vector& x) -> Vector {return -x;};

  // Jacobian of the constraints
  auto constraints_jacobian = [](const Vector& x, const Vector& /*z*/) -> Matrix {
    return -Matrix::Identity(x.size(), x.size());
  };

  // Lagrangian Hessian
  auto lagrangian_hessian = [](const Vector& /*x*/, const Vector& /*z*/) -> Matrix {
    return Matrix::Zero(0, 0); // Not used in this linear constraint example
  };

  // Create the solver
  IPsolver::Solver<TestType> solver(objective, objective_gradient, constraints, constraints_jacobian, lagrangian_hessian);
  solver.descent_direction(IPsolver::Solver<TestType>::Descent::BFGS);
  solver.tolerance(1e-4);
  solver.max_iterations(100);
  solver.verbose(false);

  // Solve the optimization problem
  Vector x_guess(Vector::Ones(2*m));
  Vector x_sol(solver.solve(x_guess));

  // Expect sparsity pattern approximately similar to true beta
  Vector w(x_sol.head(m) - x_sol.tail(m));
  REQUIRE(w.size() == m);
  for (Integer i{0}; i < m; ++i) {REQUIRE(std::abs(w(i) - beta(i)) < TestType(1.0));}
}