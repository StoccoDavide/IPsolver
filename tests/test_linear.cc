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

// Catch2 library
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/catch_template_test_macros.hpp>

// IPsolver includes
#include "IPsolver.hh"

using IPsolver::Integer;

// Logistic regression class
template<typename T>
class QuadraticProgram : public IPsolver::Problem<T> {
public:
  using typename IPsolver::Problem<T>::Vector;
  using typename IPsolver::Problem<T>::Matrix;

private:
  Matrix m_Q;
  Vector m_c;
  Matrix m_A;
  Vector m_b;

public:
  // Constructor
  QuadraticProgram(const Matrix& Q, const Vector& c, const Matrix& A, const Vector& b)
    : m_Q(Q), m_c(c), m_A(A), m_b(b) {}

  // Objective function
  T objective(const Vector& x) const override {
    return 0.5 * x.dot(this->m_Q * x) + this->m_c.dot(x);
  }

  // Gradient of the objective function
  Vector objective_gradient(const Vector& x) const override {
    return this->m_Q * x + this->m_c;
  }

  // Hessian of the objective function
  Matrix objective_hessian(const Vector& /*x*/) const override {
    return this->m_Q;
  }

  // Constraints function
  Vector constraints(const Vector& x) const override {
    return this->m_A * x - this->m_b;
  }

  // Jacobian of the constraints function
  Matrix constraints_jacobian(const Vector& /*x*/, const Vector& /*z*/) const override {
    return this->m_A;
  }

  // Hessian of the Lagrangian function
  Matrix lagrangian_hessian(const Vector& /*x*/, const Vector& /*z*/) const override {
    return this->m_Q;
  }

}; // QuadraticProgram class

// Linear regression example
TEMPLATE_TEST_CASE("Linear regression example", "[template]", double) // float, double, long double
{
  using Vector = typename IPsolver::Problem<TestType>::Vector;
  using Matrix = typename IPsolver::Problem<TestType>::Matrix;

  constexpr Integer n{2}; // Number of variables
  constexpr Integer m{5}; // Number of constraints

  // Create the dataset
  Matrix Q(2.0 * Matrix::Identity(n, n));
  Vector c(n);
  c << -2.0, -5.0;

  // Create the constraints
  Matrix A(m, n);
  A <<  1.0,  2.0,
       -1.0,  2.0,
       -1.0, -2.0,
        1.0,  0.0,
        0.0,  1.0;
  Vector b(m);
  b << 6.0, 2.0, 2.0, 3.0, 2.0;

  // Problem solution
  Vector x_guess(n);
  x_guess << 0.5, 0.5;
  Vector sol(n);
  sol << 1.4, 1.7;

  // Create the problem object
  std::unique_ptr<QuadraticProgram<TestType>> problem{
    std::make_unique<QuadraticProgram<TestType>>(Q, c, A, b)
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
    solver.tolerance(5e-5);
    solver.max_iterations(100);

    // Does not work with this example, but kept for consistency
    // SECTION("BFGS Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    //   REQUIRE(x_sol.isApprox(sol, 1e-4));
    // }

    // Does not work with this example, but kept for consistency
    // SECTION("Newton Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    //   REQUIRE(x_sol.isApprox(sol, 1e-4));
    // }

    SECTION("Steepest Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-4));
    }
  }

  // Create the quadratic program with the problem wrapper class
  SECTION("Problem Wrapper Class")
  {
    IPsolver::Solver<TestType> solver(
      problem_wrapper.objective(), problem_wrapper.objective_gradient(), problem_wrapper.objective_hessian(),
      problem_wrapper.constraints(), problem_wrapper.constraints_jacobian(), problem_wrapper.lagrangian_hessian()
    );
    solver.tolerance(5e-5);
    solver.max_iterations(100);

    // Does not work with this example, but kept for consistency
    // SECTION("BFGS Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::BFGS);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    //   REQUIRE(x_sol.isApprox(sol, 1e-4));
    // }

    // Does not work with this example, but kept for consistency
    // SECTION("Newton Descent")
    // {
    //   solver.descent(IPsolver::Solver<TestType>::Descent::NEWTON);
    //   Vector x_sol;
    //   REQUIRE(solver.solve(x_guess, x_sol));
    //   REQUIRE(x_sol.isApprox(sol, 1e-4));
    // }

    SECTION("Steepest Descent")
    {
      solver.descent(IPsolver::Solver<TestType>::Descent::STEEPEST);
      Vector x_sol;
      REQUIRE(solver.solve(x_guess, x_sol));
      REQUIRE(x_sol.isApprox(sol, 1e-4));
    }
  }
}
