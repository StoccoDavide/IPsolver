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

// Catch2 library
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/catch_template_test_macros.hpp>

// IPsolver includes
#include "IPsolver.hh"

// Define the quadratic program example
TEMPLATE_TEST_CASE("IPsolver quadratic program", "[template]", double) // float, double, long double
{
  using Integer = typename IPsolver::Integer;
  using Vector  = typename IPsolver::Solver<TestType>::Vector;
  using Matrix  = typename IPsolver::Solver<TestType>::Matrix;

  // Define the number of optimization variables and constraints
  constexpr Integer n{4};
  constexpr Integer m{3};

  // Define the Hessian of the objective function
  Matrix H(Matrix::Zero(n, n));
  H.diagonal() << 2.0, 2.0, 4.0, 2.0;
  Vector q(n);
  q << -5.0, -5.0, -21.0, 7.0;

  // Define the Hessian matrices for the constraints
  std::vector<Matrix> P(m, Matrix::Zero(n, n));
  P[0].diagonal() << 4.0, 2.0, 2.0, 0.0;
  P[1].diagonal() << 2.0, 2.0, 2.0, 2.0;
  P[2].diagonal() << 2.0, 4.0, 2.0, 4.0;

  // Constraints
  std::vector<Vector> r(m, Vector::Zero(n));
  r[0] <<  2.0, -1.0,  0.0, -1.0;
  r[1] <<  1.0, -1.0,  1.0, -1.0;
  r[2] << -1.0,  0.0,  0.0, -1.0;

  // Right-hand side vector
  Vector b(m);
  b << 5.0, 8.0, 10.0;

  // Objective function function
  auto objective = [&H, &q](const Vector& x) -> TestType {
    return 0.5 * (x.transpose() * H * x).value() + q.dot(x);
  };

  // Gradient of the objective function
  auto objective_gradient = [&H, &q](const Vector& x) -> Vector {
    return H * x + q;
  };

  // Hessian of the objective function
  auto objective_hessian = [&H](const Vector& /*x*/) -> Matrix {
    return H;
  };

  // Constraints function
  auto constraints = [&P, &r, &b](const Vector& x) -> Vector
  {
    Vector c(b.size());
    for (Integer i{0}; i < b.size(); ++i) {
      c(i) = 0.5 * x.transpose() * P[i] * x + r[i].dot(x) - b[i];
    }
    return c;
  };

  // Jacobian of the constraints function
  auto constraints_jacobian = [&P, &r, &b](const Vector& x, const Vector& /*z*/) -> Matrix
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(b.size())};
    Matrix J(m, n);
    for (Integer i{0}; i < m; ++i) {
      J.row(i) = (P[i] * x + r[i]).transpose();
    }
    return J;
  };

  // Hessian of the Lagrangian function
  auto lagrangian_hessian = [&P, &b](const Vector& x, const Vector& z) -> Matrix
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(b.size())};
    Matrix W(Matrix::Zero(n, n));
    for (Integer i{0}; i < m; ++i) {
      W += z(i) * P[i];
    }
    return W;
  };

  // Create the solver
  IPsolver::Solver<TestType> solver(
    objective, objective_gradient, objective_hessian, constraints, constraints_jacobian, lagrangian_hessian
  );
  solver.tolerance(1e-6);
  solver.max_iterations(100);
  solver.verbose(false);

  // Problem solution
  Vector x_guess(Vector::Zero(n));
  Vector sol(n);
  sol << 0.0, 1.0, 2.0, -1.0;

  // Solve the optimization problem with BFGS descent
  SECTION("BFGS Descent")
  {
    solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
    Vector x_sol(solver.solve(x_guess));
    REQUIRE(x_sol.isApprox(sol, 1e-6));
  }

  // Solve the optimization problem with Newton descent
  SECTION("NEWTON Descent")
  {
    solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
    Vector x_sol(solver.solve(x_guess));
    REQUIRE(x_sol.isApprox(sol, 1e-6));
  }
}
