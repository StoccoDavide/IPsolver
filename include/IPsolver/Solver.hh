/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 * Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                      *
 *                                                                                               *
 * The IPsolver project is distributed under the MIT License.                                    *
 *                                                                                               *
 * Davide Stocco                                                               Enrico Bertolazzi *
 * University of Trento                                                     University of Trento *
 * e-mail: davide.stocco@unitn.it                             e-mail: enrico.bertolazzi@unitn.it *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#ifndef INCLUDE_IPSOLVER_SOLVER_HH
#define INCLUDE_IPSOLVER_SOLVER_HH

#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

namespace IPsolver {

  template<typename Real>
  class Solver
  {
  public:
    using Descent = enum class Descent : Integer {
      NEWTON = 0, /**< Use Newton's method for descent direction */
      BFGS   = 1  /**< Use BFGS method for descent direction */
    }; /**< Descent direction enumeration */

    using Vector = Eigen::VectorXd; /**< Vector type (Eigen dense dynamic matrix) */
    using Matrix = Eigen::MatrixXd; /**< Matrix type (Eigen dense dynamic matrix) */

    using ObjectiveFunc           = std::function<Real(const Vector&)>;   /**< Objective function type */
    using ObjectiveGradientFunc   = std::function<Vector(const Vector&)>; /**< Gradient function type */
    using ObjectiveHessianFunc    = std::function<Matrix(const Vector&)>; /**< Hessian function type */
    using ConstraintsFunc         = std::function<Vector(const Vector&)>; /**< Constraints function type */
    using ConstraintsJacobianFunc  = std::function<Matrix(const Vector&, const Vector&)>; /**< Jacobian function type */
    using LagrangianHessianFunc   = std::function<Matrix(const Vector&, const Vector&)>; /**< Hessian of the Lagrangian function type */

  private:
    ObjectiveFunc           m_objective{nullptr};            /**< Objective function \f$ f(\mathbf{x}) \f$ */
    ObjectiveGradientFunc   m_objective_gradient{nullptr};   /**< Gradient of the objective function \f$ \nabla f(\mathbf{x}) \f$ */
    ObjectiveHessianFunc    m_objective_hessian{nullptr};    /**< Hessian of the objective function \f$ \nabla^2 f(\mathbf{x}) \f$ */
    ConstraintsFunc         m_constraints{nullptr};          /**< Constraints function \f$ g(\mathbf{x}) \f$ */
    ConstraintsJacobianFunc m_constraints_jacobian{nullptr}; /**< Jacobian of the constraints \f$ J(\mathbf{x}) \f$ */
    LagrangianHessianFunc   m_lagrangian_hessian{nullptr};  /**< Hessian of the Lagrangian \f$ W(\mathbf{x}, \mathbf{z}) \f$ */

    Descent m_descent{Descent::NEWTON}; /**< Descent direction method */
    Real    m_tolerance{1e-6};          /**< Tolerance for convergence */
    Integer m_max_iterations{100};      /**< Maximum number of iterations */
    bool    m_verbose{false};           /**< Verbose output */

    // Algorithm parameters
    Real m_epsilon{1e-8};    /**< Small constant to avoid numerical issues */
    Real m_sigma_max{0.5};   /**< Maximum value for the centering parameter */
    Real m_eta_max{0.25};    /**< Maximum value for the step size */
    Real m_mu_min{1e-9};     /**< Minimum value for the barrier parameter */
    Real m_alpha_max{0.995}; /**< Maximum value for the line search parameter */
    Real m_alpha_min{1e-6};  /**< Minimum value for the line search parameter */
    Real m_beta{0.75};       /**< Value for the backtracking line search */
    Real m_tau{0.01};        /**< Parameter for the sufficient decrease condition */

  public:
    /**
     * \brief Default constructor for the IPSolver class.
     *
     * Initializes the solver with default values for the objective, gradient, constraints, and Jacobian functions.
     */
    Solver() {};

    /**
     * \brief Constructor for the IPSolver class.
     *
     * Initializes the solver with the provided objective, gradient, constraints, and Jacobian functions.
     * \param[in] objective Objective function handle.
     * \param[in] objective_gradient Gradient of the objective function handle.
     * \param[in] constraints Constraints function handle.
     * \param[in] constraints_jacobian Jacobian of the constraints function handle.
     * \param[in] lagrangian_hessian Hessian of the Lagrangian function handle.
     * \warning The default descent direction is set to BFGS (approximation of the Hessian).
     */
    Solver(ObjectiveFunc objective, ObjectiveGradientFunc objective_gradient, ConstraintsFunc constraints,
      ConstraintsJacobianFunc constraints_jacobian, LagrangianHessianFunc lagrangian_hessian)
      : m_objective(std::move(objective)), m_objective_gradient(std::move(objective_gradient)),
        m_constraints(std::move(constraints)), m_constraints_jacobian(std::move(constraints_jacobian)),
        m_lagrangian_hessian(std::move(lagrangian_hessian)), m_descent(Descent::BFGS)
    {
      #define CMD "IPsolver::Solver::Solver(...): "

      IPSOLVER_ASSERT(this->m_objective,
        CMD "objective function must not be null");
      IPSOLVER_ASSERT(this->m_objective_gradient,
        CMD "gradient of the objective function must not be null");
      IPSOLVER_ASSERT(this->m_constraints,
        CMD "constraints function must not be null");
      IPSOLVER_ASSERT(this->m_constraints_jacobian,
        CMD "jacobian of the constraints function must not be null");
      IPSOLVER_ASSERT(this->m_lagrangian_hessian,
        CMD "lagrangian hessian function must not be null");

      #undef CMD
    }

    /**
     * \brief Constructor for the IPSolver class (with Hessian).
     *
     * Initializes the solver with the provided objective, gradient, constraints, Jacobian, and Hessian functions.
     * \param[in] objective Objective function handle.
     * \param[in] objective_gradient Gradient of the objective function handle.
     * \param[in] objective_hessian Hessian of the objective function handle.
     * \param[in] constraints Constraints function handle.
     * \param[in] constraints_jacobian Jacobian of the constraints function handle.
     * \param[in] lagrangian_hessian Hessian of the Lagrangian function handle.
     * \warning The default descent direction is set to Newton (exact Hessian).
     */
    Solver(ObjectiveFunc objective, ObjectiveGradientFunc objective_gradient, ObjectiveHessianFunc objective_hessian,
      ConstraintsFunc constraints, ConstraintsJacobianFunc constraints_jacobian, LagrangianHessianFunc lagrangian_hessian)
      : m_objective(std::move(objective)), m_objective_gradient(std::move(objective_gradient)),
        m_objective_hessian(std::move(objective_hessian)), m_constraints(std::move(constraints)),
        m_constraints_jacobian(std::move(constraints_jacobian)), m_lagrangian_hessian(std::move(lagrangian_hessian)),
        m_descent(Descent::NEWTON)
    {
      #define CMD "IPsolver::Solver::Solver(...): "

      IPSOLVER_ASSERT(this->m_objective,
        CMD "objective function must not be null");
      IPSOLVER_ASSERT(this->m_objective_gradient,
        CMD "gradient of the objective function must not be null");
      IPSOLVER_ASSERT(this->m_objective_hessian,
        CMD "hessian of the objective function must not be null");
      IPSOLVER_ASSERT(this->m_constraints,
        CMD "constraints function must not be null");
      IPSOLVER_ASSERT(this->m_constraints_jacobian,
        CMD "jacobian of the constraints function must not be null");
      IPSOLVER_ASSERT(this->m_lagrangian_hessian,
        CMD "lagrangian hessian function must not be null");

      #undef CMD
    }

    /**
     * \brief Deleted copy constructor.
     *
     * This class is not copyable.
     */
    Solver(const Solver&) = delete;

    /**
     * \brief Deleted assignment operator.
     *
     * This class is not assignable.
     */
    Solver& operator=(const Solver&) = delete;

    /**
     * \brief Deleted move constructor.
     *
     * This class is not movable.
     */
    Solver(Solver&&) = delete;

    /**
     * \brief Deleted move assignment operator.
     *
     * This class is not movable.
     */
    Solver& operator=(Solver&&) = delete;

    /**
     * \brief Destructor for the IPSolver class.
     *
     * Cleans up resources used by the solver.
     */
    ~Solver() = default;

    /**
     * \brief Sets the objective function for the solver.
     *
     * This method allows the user to specify the objective function to be minimized.
     * \param[in] objective The objective function to set.
     */
    void objective(ObjectiveFunc objective) {
      IPSOLVER_ASSERT(objective, "IPsolver::Solver::objective(...): objective function must not be null");
      this->m_objective = std::move(objective);
    }

    /**
     * \brief Sets the descent direction for the solver.
     *
     * This method allows the user to specify whether to use the Newton method
     * or the BFGS method for computing the descent direction.
     * \param[in] direction The descent direction to use.
     */
    void descent_direction(Descent direction) {this->m_descent = direction;}

    /**
     * \brief Gets the current descent direction.
     * \return The current descent direction.
     */
    Descent descent_direction() const {return this->m_descent;}

    /**
     * \brief Sets the maximum number of iterations for the solver.
     *
     * This method allows the user to specify the maximum number of iterations
     * the solver will perform before stopping.
     *
     * \param[in] max_iterations The maximum number of iterations.
     */
    void max_iterations(Integer max_iterations) {
      IPSOLVER_ASSERT(max_iterations > 0,
        "IPsolver::Solver::max_iterations(...): max_iterations must be positive");
      this->m_max_iterations = max_iterations;
    }

    /**
     * \brief Gets the current maximum number of iterations.
     * \return The current maximum number of iterations.
     */
    Integer max_iterations() const {return this->m_max_iterations;}

    /**
     * \brief Sets the convergence tolerance for the solver.
     *
     * This method allows the user to specify the tolerance for convergence.
     * The solver will stop when the residuals are below this tolerance.
     *
     * \param[in] tolerance The convergence tolerance.
     */
    void tolerance(Real tolerance) {
      IPSOLVER_ASSERT(tolerance > 0.0,
        "IPsolver::Solver::tolerance(...): tolerance must be positive");
      this->m_tolerance = tolerance;
    }

    /**
     * \brief Sets the verbosity of the solver.
     *
     * This method allows the user to enable or disable verbose output during the solving process.
     * \param[in] verbose If true, enables verbose output; otherwise, disables it.
     */
    void verbose(bool verbose) {this->m_verbose = verbose;}

    /**
     * \brief Gets the current verbosity setting.
     * \return The current verbosity setting.
     */
    bool verbose() const {return this->m_verbose;}

    /**
     * \brief Solves the optimization problem using the interior-point method.
     *
     * This method implements the interior-point algorithm to solve the optimization problem defined
     * by the objective function, constraints, and their respective gradients and Jacobians.
     * \param[in] x_guess Initial guess for the optimization variables.
     * \return The optimal solution vector.
     */
    Vector solve(const Vector& x_guess)
    {
      Vector x(x_guess);
      Vector c(this->m_constraints(x));
      Matrix J, W;
      Integer n{static_cast<Integer>(x.size())};
      Integer m{static_cast<Integer>(c.size())};
      Integer nv{n + m};

      Vector z(Vector::Ones(m));
      Matrix B(Matrix::Identity(n, n));

      Vector g_old;
      Vector p_x;

      if (this->m_verbose) {
        std::cout << "i, f(x), lg(mu), sigma, ||r_x||, ||r_c||, alpha, #ls" << std::endl;
      }

      Real alpha{0.0};
      Integer ls{0};

      for (Integer iter{0}; iter < m_max_iterations; ++iter) {
        Real f{this->m_objective(x)};
        c = this->m_constraints(x);
        J = this->m_constraints_jacobian(x, z);
        W = this->m_lagrangian_hessian(x, z);

        Vector g;
        if (this->m_descent == Descent::NEWTON) {
          g = this->m_objective_gradient(x);
          B = this->m_objective_hessian(x);
        } else {
          g = this->m_objective_gradient(x);
        }

        Vector r_x{g + J.transpose() * z};
        Vector r_c{c.array() * z.array()};
        Vector r_0(nv);
        r_0 << r_x, r_c;

        Real norm_r0{static_cast<Real>(r_0.norm())};

        Real eta{std::min(this->m_eta_max, norm_r0 / nv)};
        Real sigma{std::min(this->m_sigma_max, std::sqrt(norm_r0 / nv))};
        Real duality_gap{static_cast<Real>(-c.dot(z))};
        Real mu{std::max(this->m_mu_min, sigma * duality_gap / m)};

        if (this->m_verbose) {
          std::cout << iter+1 << ", " << f << ", " << std::log10(mu) << ", " << sigma << ", "
            << r_x.norm() << ", " << r_c.norm() << ", " << alpha << ", " << ls << std::endl;
        }

        if (norm_r0 / nv < this->m_tolerance) {break;}

        if (this->m_descent == Descent::BFGS && iter > 1) {
          Vector y(g - g_old);
          B = this->bfgs_update(B, alpha * p_x, y);
        }

        Vector c_epsilon(c.array() - this->m_epsilon);
        Matrix S(z.array() / c_epsilon.array());
        Matrix S_diag(S.asDiagonal());

        Vector g_b(g - mu * J.transpose() * (1.0 / c_epsilon.array()).matrix());

        // Solve (B + W - J' * S * J) * px = -gb
        Matrix H(B + W - J.transpose() * S_diag * J);
        p_x = H.ldlt().solve(-g_b);

        Vector p_z(-(z + mu * (1.0 / c_epsilon.array()).matrix() + S_diag * J * p_x));

        alpha = this->m_alpha_max;
        for (Integer i{0}; i < m; ++i) {
          if (p_z[i] < 0) {alpha = std::min<Real>(alpha, this->m_alpha_max * (z[i] / (-p_z[i])));}
        }

        Real psi{this->merit(z, f, c, mu)};
        Real dpsi{this->grad_merit(z, p_x, p_z, g, c, J, mu)};
        ls = 0;

        while (true) {
          ls++;
          Vector x_new(x + alpha * p_x);
          Vector z_new(z + alpha * p_z);
          Real f_new{this->m_objective(x_new)};
          Vector c_new(this->m_constraints(x_new));
          Real psi_new{this->merit(z_new, f_new, c_new, mu)};

          bool allFeasible{(c_new.array() <= 0.0).all()};
          if (allFeasible && psi_new < psi + this->m_tau * eta * alpha * dpsi) {
            x = x_new;
            z = z_new;
            g_old = g;
            break;
          }
          alpha *= this->m_beta;
          if (alpha < this->m_alpha_min) {
            IPSOLVER_ERROR("IPsolver::Solver::solve(...): line search step size too small");
          }
        }
      }
      return x;
    }

private:
    /**
     * \brief Computes the merit function.
     *
     * This function computes the merit function
     * \f[
     *   \psi(\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) - c(\mathbf{z})^\top \mathbf{z}
     *    -\mu \sum\log(\mathbf{c}^2 \mathbf{z} + \epsilon)
     * \f]
     * It is used to evaluate the quality of the current solution in terms of both the objective function and the constraints.
     * \param[in] z Dual variable vector.
     * \param[in] f Objective function value at x.
     * \param[in] c Constraints vector at x.
     * \param[in] mu Barrier parameter.
     * \return The computed merit value.
     */
    Real merit(const Vector& z, Real f, const Vector& c, Real mu)
    {
      return f - c.dot(z) - mu * ((c.array().square() * z.array() + this->m_epsilon).log().sum());
    }

    /**
     * \brief Computes the directional derivative of the merit function.
     *
     * This function computes the directional derivative of the merit function with respect to the
     * primal and dual variables.
     * \param[in] z Dual variable vector.
     * \param[in] p_x Directional derivative of the primal variable.
     * \param[in] p_z Directional derivative of the dual variable.
     * \param[in] g Gradient of the objective function at x.
     * \param[in] c Constraints vector at x.
     * \param[in] J Jacobian matrix of the constraints.
     * \param[in] mu Barrier parameter.
     * \return The computed directional derivative of the merit function.
     */
    Real grad_merit(const Vector& z, const Vector& p_x, const Vector& p_z, const Vector& g, const Vector& c,
      const Matrix& J, Real mu)
    {
      Vector term1(g - J.transpose() * z - 2.0 * mu * J.transpose() * (1.0 / (c.array() - this->m_epsilon)).matrix());
      Vector term2(c + mu * (1.0 / (z.array() + this->m_epsilon)).matrix());
      return p_x.dot(term1) - p_z.dot(term2);
    }

    /**
     * \brief Browder-Broyden-Fletcher-Goldfarb-Shanno (BFGS) update for the Hessian approximation.
     *
     * This method updates the Hessian approximation using the Browder-Broyden-Fletcher-Goldfarb-Shanno
     * (BFGS) formula*
     * \f[
     *   \mathbf{B}_{k+1} = \mathbf{B}_{k} - \displaystyle\frac{(\mathbf{B}_{k}\mathbf{s}_{k})(
     *     \mathbf{B}_{k}\mathbf{s}_{k}^\top)}{\mathbf{s}_{k}^\top \mathbf{B}_{k}\mathbf{s}_{k}} +
     *     \displaystyle\frac{\mathbf{y}\mathbf{y}^\top)}{\mathbf{y}^\top\mathbf{s}_{k}}
     * \f]
     * where \f$B\f$ is the current Hessian approximation, \f$s\f$ is the step taken,
     * and \f$y\f$ is the gradient difference.
     * \note The condition \f$y^\top s > 0\f$ must be satisfied for the update to be valid.
     * \param[in] B Current Hessian approximation.
     * \param[in] s Step taken (s_{k} = x_{k+1} - x_{k}).
     * \param[in] y Gradient difference (g_new - g).
     * \return Updated Hessian approximation.
     */
    Matrix bfgs_update(const Matrix& B, const Vector& s, const Vector& y)
    {
      if (y.dot(s) <= 0) {IPSOLVER_ERROR("BFGS update condition yᵀs > 0 not satisfied");}
      Vector Bs(B * s);
      return B - (Bs * Bs.transpose()) / (s.dot(Bs)) + (y * y.transpose()) / (y.dot(s));
    }

  }; // class IPSolver

} // namespace IPsolver

#endif /* INCLUDE_IPSOLVER_SOLVER_HH */
