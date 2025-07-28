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
TEMPLATE_TEST_CASE("IPsolver logistic regression example", "[template]", double) { // float, double, long double
  using Integer = typename IPsolver::Integer;
  using Vector  = typename IPsolver::Solver<TestType>::Vector;
  using Matrix  = typename IPsolver::Solver<TestType>::Matrix;

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

  // Objective function
  auto objective = [&P, &y, &logit](const Vector& x) -> TestType {
    Vector u(logit(P * x));
    return -(y.array() * u.array().log() + (1.0 - y.array()) * (1.0 - u.array()).log()).sum()
      + lambda * x.sum();
  };

  // Gradient of the objective function
  auto objective_gradient = [&P, &y, &lambda, &logit](const Vector& x) -> Vector {
    Vector u(logit(P * x));
    return ((-P.transpose() * (y - u)).array() + lambda).matrix();
  };

  // Hessian of the objective function
  auto objective_hessian = [&P, &logit](const Vector& x) -> Matrix {
    Vector u(logit(P * x));
    Matrix U((u.array() * (1.0 - u.array())).matrix().asDiagonal());
    return P.transpose() * U * P;
  };

  // Constraints function
  auto constraints = [](const Vector& x) -> Vector {
    return -x;
  };

  // Jacobian of the constraints function
  auto constraints_jacobian = [](const Vector& x, const Vector& /*z*/) -> Matrix {
    return -Matrix::Identity(x.size(), x.size());
  };

  // Hessian of the Lagrangian function
  auto lagrangian_hessian = [](const Vector& x, const Vector& /*z*/) -> Matrix {
    return Matrix::Zero(x.size(), x.size());
  };

  // Create the solver
  IPsolver::Solver<TestType> solver(
    objective, objective_gradient, objective_hessian, constraints, constraints_jacobian, lagrangian_hessian
  );
  solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
  solver.tolerance(1e-4);
  solver.max_iterations(100);
  solver.verbose(true);

  // Solve the optimization problem
  Vector x_guess(Vector::Ones(2*m));
  Vector x_sol(solver.solve(x_guess));

}