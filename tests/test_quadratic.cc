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

using IPsolver::Integer;

// Quadratic program class
template<typename T>
class QuadraticProgram : public IPsolver::Problem<T>
{
public:
  using typename IPsolver::Problem<T>::Vector;
  using typename IPsolver::Problem<T>::Matrix;

private:
  Matrix m_H;              // Hessian of the objective function
  Vector m_q;              // Gradient of the objective function
  std::vector<Matrix> m_P; // Hessian matrices for the constraints
  std::vector<Vector> m_r; // Gradient vectors for the constraints
  Vector m_b;              // Right-hand side vector for the constraints

public:
  // Constructor
  QuadraticProgram(const Matrix& H, const Vector& q, const std::vector<Matrix>& P,
    const std::vector<Vector>& r, const Vector& b)
    : m_H(H), m_q(q), m_P(P), m_r(r), m_b(b) {}

  // Objective function
  T objective(const Vector& x) const override
  {
    return 0.5 * (x.transpose() * this->m_H * x).value() + this->m_q.dot(x);
  }

  // Gradient of the objective function
  Vector objective_gradient(const Vector& x) const override
  {
    return this->m_H * x + this->m_q;
  }

  // Hessian of the objective function
  Matrix objective_hessian(const Vector& /*x*/) const override
  {
    return this->m_H;
  }

  // Constraints function
  Vector constraints(const Vector& x) const override
  {
    Vector c(this->m_b.size());
    for (Integer i{0}; i < this->m_b.size(); ++i) {
      c(i) = 0.5 * x.transpose() * this->m_P[i] * x + this->m_r[i].dot(x) - this->m_b(i);
    }
    return c;
  }

  // Jacobian of the constraints function
  Matrix constraints_jacobian(const Vector& x, const Vector& /*z*/) const override
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(this->m_b.size())};
    Matrix J(m, n);
    for (Integer i{0}; i < m; ++i) {
      J.row(i) = (this->m_P[i] * x + this->m_r[i]).transpose();
    }
    return J;
  }

  // Hessian of the Lagrangian function
  Matrix lagrangian_hessian(const Vector& x, const Vector& z) const override
  {
    Integer n{static_cast<Integer>(x.size())};
    Integer m{static_cast<Integer>(m_b.size())};
    Matrix W(Matrix::Zero(n, n));
    for (Integer i{0}; i < m; ++i) {
      W += z(i) * m_P[i];
    }
    return W;
  }

}; // QuadraticProgram class

// Define the quadratic program example
TEMPLATE_TEST_CASE("Quadratic program", "[template]", double) // float, double, long double
{
  using Vector = typename IPsolver::Solver<TestType>::Vector;
  using Matrix = typename IPsolver::Solver<TestType>::Matrix;

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

  // Problem solution
  Vector x_guess(Vector::Zero(n));
  Vector sol(n);
  sol << 0.0, 1.0, 2.0, -1.0;

  // Create the problem object
  std::unique_ptr<QuadraticProgram<TestType>> problem{
    std::make_unique<QuadraticProgram<TestType>>(H, q, P, r, b)
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

    IPsolver::Solver<TestType> solver(std::move(problem));
    solver.tolerance(1e-6);
    solver.max_iterations(100);

    SECTION("BFGS Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-6));
    }

    SECTION("Newton Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-6));
    }

    SECTION("Steepest Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-6));
    }
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

    SECTION("BFGS Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-6));
    }

    SECTION("Newton Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-6));
    }

    SECTION("Steepest Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-6));
    }
  }
}
