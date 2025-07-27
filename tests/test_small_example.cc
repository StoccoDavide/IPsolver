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

// STL includes
#include <vector>

// Catch2 library
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/catch_template_test_macros.hpp>

// IPsolver includes
#include "IPsolver.hh"

// Define the number of optimization variables and constraints
constexpr int n{4};
constexpr int m{3};

// Define the template test case for the small example
TEMPLATE_TEST_CASE("IPsolver small example", "[template]", double, float)
{
  using Integer = typename IPsolver::Integer;
  using Vector  = typename IPsolver::Solver<TestType>::Vector;
  using Matrix  = typename IPsolver::Solver<TestType>::Matrix;

  // Define the Hessian of the objective function
  Matrix H(Matrix::Zero(n, n));
  H.diagonal() << 2, 2, 4, 2;
  Vector q(n);
  q << -5, -5, -21, 7;

  // Define the Hessian matrices for the constraints
  std::vector<Matrix> P(m, Matrix::Zero(n, n));
  P[0].diagonal() << 4, 2, 2, 0;
  P[1].diagonal() << 2, 2, 2, 2;
  P[2].diagonal() << 2, 4, 2, 4;

  // Constraints rᵢ(x) = 0.5 * xᵀ P[i] x + r[i]ᵀ x - b[i]
  std::vector<Vector> r(m, Vector::Zero(n));
  r[0] <<  2, -1,  0, -1;
  r[1] <<  1, -1,  1, -1;
  r[2] << -1,  0,  0, -1;

  // Right-hand side vector
  Vector b(m);
  b << 5, 8, 10;

  // Objective function
  auto objective = [H, q](const Vector& x) -> TestType {
    return 0.5 * x.transpose() * H * x + q.dot(x);
  };

  // Gradient of the objective
  auto gradient = [H, q](const Vector& x) -> Vector {
    return H * x + q;
  };

  // Constraints ci(x) = 0.5 * xᵀ P[i] x + r[i]ᵀ x - b[i] < 0
  auto constraints = [P, r, b](const Vector& x) -> Vector
  {
    Vector c(b.size());
    for (Integer i{0}; i < b.size(); ++i) {
      c(i) = 0.5 * x.transpose() * P[i] * x + r[i].dot(x) - b[i];
    }
    return c;
  };

  // Jacobian and constraint Hessian
  auto jacobian = [P, r, b](const Vector& x, const Vector& z) -> std::pair<Matrix, Matrix>
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(b.size())};
    Matrix J(m, n);
    Matrix W = Matrix::Zero(n, n); // W = sum z_i * P[i]

    for (Integer i{0}; i < m; ++i) {
      J.row(i) = x.transpose() * P[i] + r[i].transpose();
      W += z(i) * P[i];
    }
    return {J, W};
  };

  // Initial guess
  Vector x_guess(Vector::Zero(n));

  // Create solver
  IPsolver::Solver<TestType> solver(objective, gradient, constraints, jacobian);
  solver.descent_direction(IPsolver::Solver<TestType>::Descent::BFGS);
  solver.tolerance(1e-6);
  solver.max_iterations(100);
  solver.verbose(true);

  // Solve the optimization problem
  Vector x_sol = solver.solve(x_guess);

  // Check the solution
  Vector sol(n);
  sol << 0.0, 1.0, 2.0, -1.0;
  REQUIRE(x_sol.size() == n);
  REQUIRE(x_sol.isApprox(sol, TestType(1e-6)));
}
