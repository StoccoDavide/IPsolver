/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 * Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                      *
 *                                                                                               *
 * The IPsolver project is distributed under the MIT License.                                    *
 *                                                                                               *
 * Davide Stocco                                                               Enrico Bertolazzi *
 * University of Trento                                                     University of Trento *
 * e-mail: davide.stocco@unitn.it                             e-mail: enrico.bertolazzi@unitn.it *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef INCLUDE_IPSOLVER_PROBLEM_HH
#define INCLUDE_IPSOLVER_PROBLEM_HH

#include <Eigen/Dense>

namespace IPsolver {

  /**
   * \brief Problem class for the IPsolver library
   *
   * The Problem class serves as a base class for defining optimization problems in the IPsolver library.
   * It provides a structure for encapsulating the problem's objective function, constraints, and other
   * necessary components.
   */
  template<typename Real>
  class Problem
  {
  public:
    using Vector = Eigen::Vector<Real, Eigen::Dynamic>; /**< Vector type (Eigen dense dynamic vector) */
    using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>; /**< Matrix type (Eigen dense dynamic matrix) */

    /**
     * \brief Default constructor for the Problem class.
     *
     * Initializes the problem with default values for the objective, gradient, hessian, constraints
     * and Jacobian functions.
     */
    Problem() {};

    /**
     * \brief Default copy constructor for the Problem class.
     */
    Problem(const Problem&) = default;

    /**
     * \brief Default move constructor for the Problem class.
     */
    Problem(Problem&&) = default;

    /**
     * \brief Default destructor for the Problem class.
     */
    virtual ~Problem() = default;

    /**
     * \brief Evaluate the objective function.
     * \param[in] x Primal variable vector.
     * \return The objective function.
     */
    virtual Real objective(const Vector &x) const = 0;

    /**
     * \brief Evaluate the gradient of the objective function.
     * \param[in] x Primal variable vector.
     * \return The gradient of the objective function.
     */
    virtual Vector objective_gradient(const Vector &x) const = 0;

    /**
     * \brief Evaluate the Hessian of the objective function.
     * \param[in] x Primal variable vector.
     * \return The Hessian matrix of the objective function.
     */
    virtual Matrix objective_hessian(const Vector &x) const = 0;

    /**
     * \brief Evaluate the constraints function for the problem.
     * \param[in] x Primal variable vector.
     * \return The value of the constraints function.
     */
    virtual Vector constraints(const Vector &x) const = 0;

    /**
     * \brief Evaluate the Jacobian of the constraints function.
     * \param[in] x Primal variable vector.
      * \param[in] z Dual variable vector.
     * \return The Jacobian matrix of the constraints function.
     */
    virtual Matrix constraints_jacobian(const Vector &x, const Vector &z) const = 0;

    /**
     * \brief Evaluate the Hessian of the Lagrangian function.
     * \param[in] x Primal variable vector.
     * \param[in] z Dual variable vector.
     * \return The Hessian matrix of the Lagrangian function.
     */
    virtual Matrix lagrangian_hessian(const Vector &x, const Vector &z) const = 0;

  }; // class Problem


  /**
   * \brief Wrapper class for the Problem class.
   *
   * This class is a simple wrapper around the Problem class, allowing for easy instantiation and usage.
   * It inherits from the Problem class and can be used to define specific optimization problems.
   */
  template<typename Real>
  class ProblemWrapper : public Problem<Real>
  {
  public:
    using Vector = typename Problem<Real>::Vector;
    using Matrix = typename Problem<Real>::Matrix;

    using ObjectiveFunc           = std::function<Real(const Vector&)>;                  /**< Objective function type */
    using ObjectiveGradientFunc   = std::function<Vector(const Vector&)>;                /**< Gradient function type */
    using ObjectiveHessianFunc    = std::function<Matrix(const Vector&)>;                /**< Hessian function type */
    using ConstraintsFunc         = std::function<Vector(const Vector&)>;                /**< Constraints function type */
    using ConstraintsJacobianFunc = std::function<Matrix(const Vector&, const Vector&)>; /**< Jacobian function type */
    using LagrangianHessianFunc   = std::function<Matrix(const Vector&, const Vector&)>; /**< Hessian of the Lagrangian function type */

  private:
    ObjectiveFunc           m_objective{nullptr};            /**< Objective function \f$ f(\mathbf{x}) \f$ */
    ObjectiveGradientFunc   m_objective_gradient{nullptr};   /**< Gradient of the objective function \f$ \nabla f(\mathbf{x}) \f$ */
    ObjectiveHessianFunc    m_objective_hessian{nullptr};    /**< Hessian of the objective function \f$ \nabla^2 f(\mathbf{x}) \f$ */
    ConstraintsFunc         m_constraints{nullptr};          /**< Constraints function \f$ g(\mathbf{x}) \f$ */
    ConstraintsJacobianFunc m_constraints_jacobian{nullptr}; /**< Jacobian of the constraints \f$ J(\mathbf{x}) \f$ */
    LagrangianHessianFunc   m_lagrangian_hessian{nullptr};   /**< Hessian of the Lagrangian \f$ W(\mathbf{x}, \mathbf{z}) \f$ */

  public:
    /**
     * \brief Constructor for the ProblemWrapper class (without the Hessian of the Lagrangian).
     *
     * Initializes the problem with the provided objective, gradient, constraints, and Jacobian functions.
     * \param[in] objective Objective function handle.
     * \param[in] objective_gradient Gradient of the objective function handle.
     * \param[in] constraints Constraints function handle.
     * \param[in] constraints_jacobian Jacobian of the constraints function handle.
     * \param[in] lagrangian_hessian Hessian of the Lagrangian function handle.
     * \warning The default descent direction is set to BFGS (approximation of the Hessian).
     */
    ProblemWrapper(ObjectiveFunc const &objective, ObjectiveGradientFunc const &objective_gradient,
      ConstraintsFunc const &constraints, ConstraintsJacobianFunc const &constraints_jacobian,
      LagrangianHessianFunc const &lagrangian_hessian)
      : m_objective(objective), m_objective_gradient(objective_gradient), m_constraints(constraints),
        m_constraints_jacobian(constraints_jacobian), m_lagrangian_hessian(lagrangian_hessian) {}

    /**
     * \brief Constructor for the ProblemWrapper class (with the Hessian of the Lagrangian).
     *
     * Initializes the problem with the provided objective, gradient, constraints, Jacobian, and Hessian functions.
     * \param[in] objective Objective function handle.
     * \param[in] objective_gradient Gradient of the objective function handle.
     * \param[in] objective_hessian Hessian of the objective function handle.
     * \param[in] constraints Constraints function handle.
     * \param[in] constraints_jacobian Jacobian of the constraints function handle.
     * \param[in] lagrangian_hessian Hessian of the Lagrangian function handle.
     * \warning The default descent direction is set to Newton (exact Hessian).
     */
    ProblemWrapper(ObjectiveFunc const &objective, ObjectiveGradientFunc const &objective_gradient,
      ObjectiveHessianFunc const &objective_hessian, ConstraintsFunc const &constraints,
      ConstraintsJacobianFunc const &constraints_jacobian, LagrangianHessianFunc const &lagrangian_hessian)
      : m_objective(objective), m_objective_gradient(objective_gradient),
        m_objective_hessian(objective_hessian), m_constraints(constraints),
        m_constraints_jacobian(constraints_jacobian), m_lagrangian_hessian(lagrangian_hessian) {}

    /**
     * \brief Default destructor for the ProblemWrapper class.
     */
    ~ProblemWrapper() {};

    /**
     * \brief Gets the objective function.
     * \return The objective function.
     */
    ObjectiveFunc objective() const {return this->m_objective;}

    /**
     * \brief Sets the objective function.
     * \param[in] objective The objective function to set.
     */
    void objective(ObjectiveFunc const &objective)
    {
      this->m_objective = objective;
    }

    /**
     * \brief Gets the gradient of the objective function.
     * \return The gradient of the objective function.
     */
    ObjectiveGradientFunc objective_gradient() const {return this->m_objective_gradient;}

    /**
     * \brief Sets the gradient of the objective function.
     * \param[in] objective_gradient The gradient of the objective function to set.
     */
    void objective_gradient(ObjectiveGradientFunc const &objective_gradient)
    {
      this->m_objective_gradient = objective_gradient;
    }

    /**
     * \brief Gets the Hessian of the objective function.
     * \return The Hessian of the objective function.
     */
    ObjectiveHessianFunc objective_hessian() const {return this->m_objective_hessian;}

    /**
     * \brief Sets the Hessian of the objective function.
     * \param[in] objective_hessian The Hessian of the objective function to set.
     */
    void objective_hessian(ObjectiveHessianFunc const &objective_hessian)
    {
      this->m_objective_hessian = objective_hessian;
    }

    /**
     * \brief Gets the constraints function.
     * \return The constraints function.
     */
    ConstraintsFunc constraints() const {return this->m_constraints;}

    /**
     * \brief Sets the constraints function.
     * \param[in] constraints The constraints function to set.
     */
    void constraints(ConstraintsFunc const &constraints)
    {
      this->m_constraints = constraints;
    }

    /**
     * \brief Gets the Jacobian of the constraints function.
     * \return The Jacobian of the constraints function.
     */
    ConstraintsJacobianFunc constraints_jacobian() const {return this->m_constraints_jacobian;}

    /**
     * \brief Sets the Jacobian of the constraints function.
     * \param[in] constraints_jacobian The Jacobian of the constraints function to set.
     */
    void constraints_jacobian(ConstraintsJacobianFunc const &constraints_jacobian)
    {
      this->m_constraints_jacobian = constraints_jacobian;
    }

    /**
     * \brief Gets the Hessian of the Lagrangian function.
     * \return The Hessian of the Lagrangian function.
     */
    LagrangianHessianFunc lagrangian_hessian() const {return this->m_lagrangian_hessian;}

    /**
     * \brief Sets the Hessian of the Lagrangian function.
     * \param[in] lagrangian_hessian The Hessian of the Lagrangian function to set.
     */
    void lagrangian_hessian(LagrangianHessianFunc const &lagrangian_hessian)
    {
      this->m_lagrangian_hessian = lagrangian_hessian;
    }

    /**
     * \brief Evaluate the objective function.
     * \param[in] x Primal variable vector.
     * \return The objective function.
     */
    Real objective(const Vector &x) const override {return this->m_objective(x);}

    /**
     * \brief Evaluate the gradient of the objective function.
     * \param[in] x Primal variable vector.
     * \return The gradient of the objective function.
     */
    Vector objective_gradient(const Vector &x) const override {return this->m_objective_gradient(x);}

    /**
     * \brief Evaluate the Hessian of the objective function.
     * \param[in] x Primal variable vector.
     * \return The Hessian matrix of the objective function.
     */
    Matrix objective_hessian(const Vector &x) const override {return this->m_objective_hessian(x);}

    /**
     * \brief Evaluate the constraints function.
     * \param[in] x Primal variable vector.
     * \return The value of the constraints function.
     */
    Vector constraints(const Vector &x) const override {return this->m_constraints(x);}

    /**
     * \brief Evaluate the Jacobian of the constraints function.
     * \param[in] x Primal variable vector.
     * \param[in] z Dual variable vector.
     * \return The Jacobian matrix of the constraints function.
     */
    Matrix constraints_jacobian(const Vector &x, const Vector &z) const override {return this->m_constraints_jacobian(x, z);}

    /**
     * \brief Evaluate the Hessian of the Lagrangian function.
     * \param[in] x Primal variable vector.
     * \param[in] z Dual variable vector.
     * \return The Hessian matrix of the Lagrangian function.
     */
    Matrix lagrangian_hessian(const Vector &x, const Vector &z) const override {return this->m_lagrangian_hessian(x, z);}

  }; // class ProblemWrapper

} // namespace IPsolver

#endif /* INCLUDE_IPSOLVER_PROBLEM_HH */
