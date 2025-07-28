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
      NEWTON   = 0, /**< Use Newton's method for descent direction */
      BFGS     = 1, /**< Use BFGS method for descent direction */
      STEEPEST = 2  /**< Use steepest descent method for descent direction */
    }; /**< Descent direction enumeration */

    using Vector = Eigen::VectorXd; /**< Vector type (Eigen dense dynamic matrix) */
    using Matrix = Eigen::MatrixXd; /**< Matrix type (Eigen dense dynamic matrix) */
    using Array  = Eigen::ArrayXd;  /**< Array type (Eigen dense dynamic array) */
    using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>; /**< Mask type for Eigen arrays */

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

    // Some algorithm parameters
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
    Solver(ObjectiveFunc const &objective, ObjectiveGradientFunc const &objective_gradient,
      ConstraintsFunc const &constraints, ConstraintsJacobianFunc const &constraints_jacobian,
      LagrangianHessianFunc const &lagrangian_hessian)
      : m_objective(objective), m_objective_gradient(objective_gradient),
        m_constraints(constraints), m_constraints_jacobian(constraints_jacobian),
        m_lagrangian_hessian(lagrangian_hessian), m_descent(Descent::BFGS) {}

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
    Solver(ObjectiveFunc const &objective, ObjectiveGradientFunc const &objective_gradient,
      ObjectiveHessianFunc const &objective_hessian, ConstraintsFunc const &constraints,
      ConstraintsJacobianFunc const &constraints_jacobian, LagrangianHessianFunc const &lagrangian_hessian)
      : m_objective(objective), m_objective_gradient(objective_gradient),
        m_objective_hessian(objective_hessian), m_constraints(constraints),
        m_constraints_jacobian(constraints_jacobian), m_lagrangian_hessian(lagrangian_hessian),
        m_descent(Descent::NEWTON) {}

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
    void objective(ObjectiveFunc objective) {this->m_objective = objective;}

    /**
     * \brief Gets the current objective function.
     * \return The current objective function.
     */
    ObjectiveFunc objective() const {return this->m_objective;}

    /**
     * \brief Sets the gradient of the objective function for the solver.
     *
     * This method allows the user to specify the gradient of the objective function.
     * \param[in] objective_gradient The gradient of the objective function to set.
     */
    void objective_gradient(ObjectiveGradientFunc objective_gradient) {
      this->m_objective_gradient = objective_gradient;
    }

    /**
     * \brief Gets the current gradient of the objective function.
     * \return The current gradient of the objective function.
     */
    ObjectiveGradientFunc objective_gradient() const {return this->m_objective_gradient;}

    /**
     * \brief Sets the Hessian of the objective function for the solver.
     *
     * This method allows the user to specify the Hessian of the objective function.
     * \param[in] objective_hessian The Hessian of the objective function to set.
     */
    void objective_hessian(ObjectiveHessianFunc objective_hessian) {
      this->m_objective_hessian = objective_hessian;
    }

    /**
     * \brief Gets the current Hessian of the objective function.
     * \return The current Hessian of the objective function.
     */
    ObjectiveHessianFunc objective_hessian() const {return this->m_objective_hessian;}

    /**
     * \brief Sets the constraints function for the solver.
     *
     * This method allows the user to specify the constraints function.
     * \param[in] constraints The constraints function to set.
     */
    void constraints(ConstraintsFunc constraints) {this->m_constraints = constraints;}

    /**
     * \brief Gets the current constraints function.
     * \return The current constraints function.
     */
    ConstraintsFunc constraints() const {return this->m_constraints;}

    /**
     * \brief Sets the Jacobian of the constraints function for the solver.
     *
     * This method allows the user to specify the Jacobian of the constraints function.
     * \param[in] constraints_jacobian The Jacobian of the constraints function to set.
     */
    void constraints_jacobian(ConstraintsJacobianFunc constraints_jacobian) {
      this->m_constraints_jacobian = constraints_jacobian;
    }

    /**
     * \brief Gets the current Jacobian of the constraints function.
     * \return The current Jacobian of the constraints function.
     */
    ConstraintsJacobianFunc constraints_jacobian() const {return this->m_constraints_jacobian;}

    /**
     * \brief Sets the Hessian of the Lagrangian function for the solver.
     *
     * This method allows the user to specify the Hessian of the Lagrangian function.
     * \param[in] lagrangian_hessian The Hessian of the Lagrangian function to set.
     */
    void lagrangian_hessian(LagrangianHessianFunc lagrangian_hessian) {
      this->m_lagrangian_hessian = lagrangian_hessian;
    }

    /**
     * \brief Gets the current Hessian of the Lagrangian function.
     * \return The current Hessian of the Lagrangian function.
     */
    LagrangianHessianFunc lagrangian_hessian() const {return this->m_lagrangian_hessian;}

    /**
     * \brief Sets the descent direction for the solver.
     *
     * This method allows the user to specify whether to use the Newton method
     * or the BFGS method for computing the descent direction.
     * \param[in] descent The descent direction to use.
     */
    void descent(Descent descent) {this->m_descent = descent;}

    /**
     * \brief Gets the current descent direction.
     * \return The current descent direction.
     */
    Descent descent() const {return this->m_descent;}

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
        "IPsolver::Solver::max_iterations(...): input value must be positive");
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
        "IPsolver::Solver::tolerance(...): input value must be positive");
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
     * \brief Sets the small constant epsilon to avoid numerical issues.
     * \param[in] epsilon The small positive constant.
     */
    void epsilon(Real epsilon) {
      IPSOLVER_ASSERT(epsilon > 0.0,
        "IPsolver::Solver::epsilon(...): input value must be positive");
      this->m_epsilon = epsilon;
    }

    /**
     * \brief Gets the current epsilon value.
     * \return The current epsilon value.
     */
    Real epsilon() const {return this->m_epsilon;}

    /**
     * \brief Sets the maximum value for the centering parameter sigma.
     * \param[in] sigma_max The maximum value for sigma (must be positive).
     */
    void sigma_max(Real sigma_max) {
      IPSOLVER_ASSERT(sigma_max > 0.0,
        "IPsolver::Solver::sigma_max(...): input value must be positive");
      this->m_sigma_max = sigma_max;
    }

    /**
     * \brief Gets the current maximum value for sigma.
     * \return The current sigma_max value.
     */
    Real sigma_max() const {return this->m_sigma_max;}

    /**
     * \brief Sets the maximum value for the step size eta.
     * \param[in] eta_max The maximum value for eta (must be positive).
     */
    void eta_max(Real eta_max) {
      IPSOLVER_ASSERT(eta_max > 0.0,
        "IPsolver::Solver::eta_max(...): input value must be positive");
      this->m_eta_max = eta_max;
    }

    /**
     * \brief Gets the current maximum value for eta.
     * \return The current eta_max value.
     */
    Real eta_max() const {return this->m_eta_max;}

    /**
     * \brief Sets the minimum value for the barrier parameter mu.
     * \param[in] mu_min The minimum value for mu (must be positive).
     */
    void mu_min(Real mu_min) {
      IPSOLVER_ASSERT(mu_min > 0.0,
        "IPsolver::Solver::mu_min(...): input value must be positive");
      this->m_mu_min = mu_min;
    }

    /**
     * \brief Gets the current minimum value for mu.
     * \return The current mu_min value.
     */
    Real mu_min() const {return this->m_mu_min;}

    /**
     * \brief Sets the maximum value for the line search parameter alpha.
     * \param[in] alpha_max The maximum value for alpha (must be positive).
     */
    void alpha_max(Real alpha_max) {
      IPSOLVER_ASSERT(alpha_max > 0.0,
        "IPsolver::Solver::alpha_max(...): input value must be positive");
      this->m_alpha_max = alpha_max;
    }

    /**
     * \brief Gets the current maximum value for alpha.
     * \return The current alpha_max value.
     */
    Real alpha_max() const {return this->m_alpha_max;}

    /**
     * \brief Sets the minimum value for the line search parameter alpha.
     * \param[in] alpha_min The minimum value for alpha (must be positive).
     */
    void alpha_min(Real alpha_min) {
      IPSOLVER_ASSERT(alpha_min > 0.0,
        "IPsolver::Solver::alpha_min(...): input value must be positive");
      this->m_alpha_min = alpha_min;
    }

    /**
     * \brief Gets the current minimum value for alpha.
     * \return The current alpha_min value.
     */
    Real alpha_min() const {return this->m_alpha_min;}

    /**
     * \brief Sets the value for the backtracking line search parameter beta.
     * \param[in] beta The value for beta (must be positive).
     */
    void beta(Real beta) {
      IPSOLVER_ASSERT(beta > 0.0,
        "IPsolver::Solver::beta(...): input value must be positive");
      this->m_beta = beta;
    }

    /**
     * \brief Gets the current value for beta.
     * \return The current beta value.
     */
    Real beta() const {return this->m_beta;}

    /**
     * \brief Sets the parameter for the sufficient decrease condition tau.
     * \param[in] tau The value for tau (must be positive).
     */
    void tau(Real tau) {
      IPSOLVER_ASSERT(tau > 0.0,
        "IPsolver::Solver::tau(...): input value must be positive");
      this->m_tau = tau;
    }

    /**
     * \brief Gets the current value for tau.
     * \return The current tau value.
     */
    Real tau() const {return this->m_tau;}


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
      #define CMD "IPsolver::Solver::solve(...): "

      IPSOLVER_ASSERT(this->m_objective,
        CMD "objective function must not be null");
      IPSOLVER_ASSERT(this->m_objective_gradient,
        CMD "gradient of the objective function must not be null");
      IPSOLVER_ASSERT(this->m_objective_hessian || this->m_descent != Descent::NEWTON,
        CMD "hessian of the objective function must not be null");
      IPSOLVER_ASSERT(this->m_constraints,
        CMD "constraints function must not be null");
      IPSOLVER_ASSERT(this->m_constraints_jacobian,
        CMD "jacobian of the constraints function must not be null");
      IPSOLVER_ASSERT(this->m_lagrangian_hessian,
        CMD "lagrangian hessian function must not be null");

      // INITIALIZATION
      // Get the number of primal variables (n), the number of constraints (m), the total number of
      // primal-dual optimization variables (nv), and initialize the Lagrange multipliers and the
      // second-order information
      Vector x(x_guess);
      Vector c(this->m_constraints(x));
      Integer n{static_cast<Integer>(x.size())};
      Integer m{static_cast<Integer>(c.size())};
      Integer nv{n + m};
      Vector z(Vector::Ones(m));
      Matrix B(Matrix::Identity(n, n));

      Vector g_old, p_x, p_z;
      if (this->m_verbose) {
        std::cout << "i, f(x), lg(mu), sigma, ||r_x||, ||r_c||, alpha, #ls" << std::endl;
      }

      // Repeat while the convergence criterion has not been satisfied, and we haven't reached the
      // maximum number of iterations
      Real alpha{0.0};
      Integer ls{0};
      for (Integer iter{0}; iter < m_max_iterations; ++iter) {

        // COMPUTE OBJECTIVE, GRADIENT, CONSTRAINTS, ETC
        // Compute the response of the objective function, the gradient of the objective, the
        // response of the inequality constraints, the Jacobian of the inequality constraints, the
        // Hessian of the Lagrangian (minus the Hessian of the objective) and, optionally, the
        // Hessian of the objective.
        Real f{this->m_objective(x)};
        c = this->m_constraints(x);
        Vector g(this->m_objective_gradient(x));
        Matrix J(this->m_constraints_jacobian(x, z));
        Matrix W(this->m_lagrangian_hessian(x, z));
        if (this->m_descent == Descent::NEWTON) {
          B = this->m_objective_hessian(x);
        }

        // Compute the responses of the unperturbed Karush-Kuhn-Tucker optimality conditions.
        Vector r_x{g + J.transpose() * z}; // Dual residual
        Vector r_c{c.array() * z.array()}; // Complementarity
        Vector r_0(nv);
        r_0 << r_x, r_c;

        // Set some parameters that affect convergence of the primal-dual interior-point method
        Real norm_r0{static_cast<Real>(r_0.norm())};
        Real eta{std::min<Real>(this->m_eta_max, norm_r0 / nv)};
        Real sigma{std::min<Real>(this->m_sigma_max, std::sqrt(norm_r0 / nv))};
        Real duality_gap{static_cast<Real>(-c.dot(z))};
        Real mu{std::max<Real>(this->m_mu_min, sigma * duality_gap / m)};

        // Print the status of the algorithm
        if (this->m_verbose) {
          std::cout << iter+1 << ", " << f << ", " << std::log10(mu) << ", " << sigma << ", "
            << r_x.norm() << ", " << r_c.norm() << ", " << alpha << ", " << ls << std::endl;
        }

        // CONVERGENCE CHECK
        // If the norm of the responses is less than the specified tolerance, we are done
        if (norm_r0 / nv < this->m_tolerance) {break;}

        // Update the BFGS approximation to the Hessian of the objective
        if (this->m_descent == Descent::BFGS && iter > 0) {
          B = this->bfgs_update(B, alpha * p_x, g - g_old);
        }

        // SOLUTION TO PERTURBED KKT SYSTEM
        // Compute the search direction of x and z
        Vector c_epsilon(c.array() - this->m_epsilon);
        Matrix S((z.array() / c_epsilon.array()).matrix().asDiagonal());
        Vector g_b(g - mu * J.transpose() * (1.0 / c_epsilon.array()).matrix());
        Matrix H(B + W - J.transpose() * S * J);
        p_x = H.ldlt().solve(-g_b);
        p_z = -(z + mu * (1.0 / c_epsilon.array()).matrix() + S * J * p_x);

        // BACKTRACKING LINE SEARCH
        // To ensure global convergence, execute backtracking line search to determine the step length
        // First, we have to find the largest step size which ensures that z remains feasible
        // Next, we perform backtracking line search
        alpha = this->m_alpha_max;
        Mask mask((z + p_z).array() < 0.0);
        if (mask.any()) {
          Array ratio(z.array() / (-p_z.array()));
          alpha = this->m_alpha_max * std::min<Real>(1.0, mask.select(ratio, 1.0).minCoeff());
        }

        // Compute the response of the merit function and the directional gradient at the current
        // point and search direction
        Real psi{this->merit(x, z, f, c, mu)};
        Real dpsi{this->grad_merit(x, z, p_x, p_z, g, c, J, mu)};
        ls = 0;
        while (true) {

          // Compute the candidate point, the constraints, and the response of the objective function
          // and merit function at the candidate point
          ++ls;
          Vector x_new(x + alpha * p_x);
          Vector z_new(z + alpha * p_z);
          f = this->m_objective(x_new);
          c = this->m_constraints(x_new);
          Real psi_new{this->merit(x_new, z_new, f, c, mu)};

          // Stop backtracking search if we've found a candidate point that sufficiently decreases
          // the merit function and satisfies all the constraints
          if ((c.array() <= 0.0).all() && psi_new < psi + this->m_tau * eta * alpha * dpsi) {
            x = x_new;
            z = z_new;
            g_old = g;
            break;
          }

          // The candidate point does not meet our criteria, so decrease the step size for 0 < β < 1.
          alpha *= this->m_beta;
          IPSOLVER_ASSERT(alpha > this->m_alpha_min, CMD "line search step size too small");
        }
      }
      return x;

      #undef CMD
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
     * \param[in] x Primal variable vector.
     * \param[in] z Dual variable vector.
     * \param[in] f Objective function value at x.
     * \param[in] c Constraints vector at x.
     * \param[in] mu Barrier parameter.
     * \return The computed merit value.
     */
    Real merit([[maybe_unused]] const Vector& x, const Vector& z, Real f, const Vector& c, Real mu)
    {
      return f - c.dot(z) - mu * ((c.array().square() * z.array() + this->m_epsilon).log().sum());
    }

    /**
     * \brief Computes the directional derivative of the merit function.
     *
     * This function computes the directional derivative of the merit function with respect to the
     * primal and dual variables.
     * \param[in] x Primal variable vector.
     * \param[in] z Dual variable vector.
     * \param[in] p_x Directional derivative of the primal variable.
     * \param[in] p_z Directional derivative of the dual variable.
     * \param[in] g Gradient of the objective function at x.
     * \param[in] c Constraints vector at x.
     * \param[in] J Jacobian matrix of the constraints.
     * \param[in] mu Barrier parameter.
     * \return The computed directional derivative of the merit function.
     */
    Real grad_merit([[maybe_unused]] const Vector& x, const Vector& z, const Vector& p_x,
      const Vector& p_z, const Vector& g, const Vector& c, const Matrix& J, Real mu)
    {
      return p_x.dot(
          g - J.transpose() * z - 2.0 * mu * J.transpose() * (1.0 / (c.array() - this->m_epsilon)).matrix()
        ) - p_z.dot(
          c + mu * (1.0 / (z.array() + this->m_epsilon)).matrix()
        );
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
      #define CMD "IPsolver::Solver::bfgs_update(...): "
      IPSOLVER_ASSERT(y.dot(s) > 0.0, CMD "update condition yᵀs > 0 not satisfied");
      Vector x(B * s);
      return B - (x * x.transpose()) / (s.dot(x)) + (y * y.transpose()) / (y.dot(s));
      #undef CMD
    }

  }; // class IPSolver

} // namespace IPsolver

#endif /* INCLUDE_IPSOLVER_SOLVER_HH */
