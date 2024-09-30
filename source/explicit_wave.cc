/* ---------------------------------------------------------------------
 *
 *
 * Authors: Svenja Schoeder and Martin Kronbichler
 *          Institute for Computational Mechanics
 *          Technical University of Munich
 *          Garching, Germany
 *          schoeder@lnm.mw.tum.de
 *          http://www.lnm.mw.tum.de
 *
 *
 * ---------------------------------------------------------------------*/


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "../include/input_parameters.h"
#include "../include/parameters.h"
#include "../include/time_integrators.h"
#include "../include/wave_equation_operations.h"

namespace HDG_WE
{
  using namespace dealii;


  template <int dim, typename Number>
  class MatrixWaveOperation
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    MatrixWaveOperation(const MatrixFree<dim, Number> &matrix_free)
      : matrix_free(matrix_free)
      , time_step(1.)
    {}

    void
    reinit(const std::vector<Material> &mats)
    {
      densities.resize(matrix_free.n_cell_batches() +
                       matrix_free.n_ghost_cell_batches());
      speeds.resize(matrix_free.n_cell_batches() +
                    matrix_free.n_ghost_cell_batches());

      for (unsigned int i = 0; i < matrix_free.n_cell_batches() +
                                     matrix_free.n_ghost_cell_batches();
           ++i)
        {
          densities[i] = 1.;
          speeds[i]    = 1.;
          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_cell_batch(i);
               ++v)
            {
              densities[i][v] =
                mats[matrix_free.get_cell_iterator(i, v)->material_id()]
                  .density;
              speeds[i][v] =
                mats[matrix_free.get_cell_iterator(i, v)->material_id()].speed;
            }
        }

      time_step = 1.;
    }

    void
    set_time_step(const double time_step_size)
    {
      this->time_step = time_step_size;
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      matrix_free.loop(&MatrixWaveOperation::apply_cell,
                       &MatrixWaveOperation::apply_face,
                       &MatrixWaveOperation::apply_boundary,
                       this,
                       dst,
                       src,
                       true,
                       MatrixFree<dim, Number>::DataAccessOnFaces::values,
                       MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    void
    apply_inverse_mass_matrix(VectorType &dst, const VectorType &src) const
    {
      FEEvaluation<dim, -1, 0, dim + 1, Number> phi(matrix_free, 0, 0, 0);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 1, Number>
        mass_inv(phi);

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          mass_inv.apply(phi.begin_dof_values(), phi.begin_dof_values());
          phi.set_dof_values(dst);
        }
    }

  private:
    const MatrixFree<dim, Number>         &matrix_free;
    AlignedVector<VectorizedArray<Number>> densities;
    AlignedVector<VectorizedArray<Number>> speeds;
    Number                                 time_step;

    void
    apply_cell(const MatrixFree<dim, Number>               &data,
               VectorType                                  &dst,
               const VectorType                            &src,
               const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim, Number> phi_v(data, 0, 0, 0);
      FEEvaluation<dim, -1, 0, 1, Number>   phi_p(data, 0, 0, dim);
      const Number factor_time = time_step == 0 ? 0. : 1. / time_step;

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          // It is faster to evaluate values of the vector-valued velocity and
          // gradients of the scalar pressure than divergence of velocity and
          // values of pressure
          phi_v.reinit(cell);
          phi_v.gather_evaluate(src, EvaluationFlags::values);

          phi_p.reinit(cell);
          phi_p.gather_evaluate(src,
                                EvaluationFlags::values |
                                  EvaluationFlags::gradients);

          const VectorizedArray<Number> rho     = this->densities[cell];
          const VectorizedArray<Number> c_sq_rho_inv =
            1. / (this->speeds[cell] * this->speeds[cell] * rho);

          for (const unsigned int q : phi_p.quadrature_point_indices())
            {
              const VectorizedArray<Number> p_val = phi_p.get_value(q);
              const Tensor<1, dim, VectorizedArray<Number>> p_grad =
                phi_p.get_gradient(q);
              const Tensor<1, dim, VectorizedArray<Number>> v_val =
                phi_v.get_value(q);

              phi_p.submit_gradient(-v_val, q);
              phi_p.submit_value(c_sq_rho_inv * factor_time * p_val, q);
              phi_v.submit_value(rho * factor_time * v_val + p_grad, q);
            }

          phi_p.integrate_scatter(EvaluationFlags::values |
                                    EvaluationFlags::gradients,
                                  dst);
          phi_v.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    apply_face(const MatrixFree<dim, Number>               &data,
               VectorType                                  &dst,
               const VectorType                            &src,
               const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // There is some overhead in the methods in FEEvaluation, so it is faster
      // to combine pressure and velocity in the same object and just combine
      // them at the level of quadrature points
      FEFaceEvaluation<dim, -1, 0, dim + 1, Number> phi(data, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim + 1, Number> phi_neighbor(
        data, false, 0, 0, 0);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          phi.reinit(face);
          phi.gather_evaluate(src, EvaluationFlags::values);
          const VectorizedArray<Number> rho_plus =
            phi.read_cell_data(densities);
          const VectorizedArray<Number> rho_inv_plus = 1. / rho_plus;
          const VectorizedArray<Number> c_plus = phi.read_cell_data(speeds);
          const VectorizedArray<Number> c_sq_rho_plus =
            c_plus * c_plus * rho_plus;
          const VectorizedArray<Number> tau_plus = 1. / (c_plus * rho_plus);

          phi_neighbor.reinit(face);
          phi_neighbor.gather_evaluate(src, EvaluationFlags::values);
          const VectorizedArray<Number> rho_minus =
            phi_neighbor.read_cell_data(densities);
          const VectorizedArray<Number> rho_inv_minus = 1. / rho_minus;
          const VectorizedArray<Number> c_minus =
            phi_neighbor.read_cell_data(speeds);
          const VectorizedArray<Number> c_sq_rho_minus =
            c_minus * c_minus * rho_minus;
          const VectorizedArray<Number> tau_minus = 1. / (c_minus * rho_minus);

          const VectorizedArray<Number> tau_inv = 1. / (tau_plus + tau_minus);

          AssertDimension(phi.n_q_points, data.get_n_q_points_face(0));

          for (const unsigned int q : phi.quadrature_point_indices())
            {
              Tensor<1, dim + 1, VectorizedArray<Number>> val_plus =
                phi.get_value(q);
              Tensor<1, dim + 1, VectorizedArray<Number>> val_minus =
                phi_neighbor.get_value(q);
              Tensor<1, dim, VectorizedArray<Number>> normal =
                phi.get_normal_vector(q);
              VectorizedArray<Number> normal_v_plus = val_plus[0] * normal[0];
              VectorizedArray<Number> normal_v_minus =
                -val_minus[0] * normal[0];
              for (unsigned int d = 1; d < dim; ++d)
                {
                  normal_v_plus += val_plus[d] * normal[d];
                  normal_v_minus -= val_minus[d] * normal[d];
                }

              VectorizedArray<Number> lambda =
                tau_inv *
                (normal_v_plus + normal_v_minus + tau_plus * val_plus[dim] +
                 tau_minus * val_minus[dim]);
              VectorizedArray<Number> pres_diff_plus =
                (val_plus[dim] - lambda) * rho_inv_plus;
              VectorizedArray<Number> pres_diff_minus =
                (val_minus[dim] - lambda) * rho_inv_minus;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  val_plus[d]  = -pres_diff_plus * normal[d];
                  val_minus[d] = pres_diff_minus * normal[d];
                }
              val_plus[dim] =
                c_sq_rho_plus *
                (normal_v_plus - tau_plus * (lambda - val_plus[dim]));
              val_minus[dim] =
                c_sq_rho_minus *
                (normal_v_minus - tau_minus * (lambda - val_minus[dim]));

              phi.submit_value(val_plus, q);
              phi_neighbor.submit_value(val_minus, q);
            }

          phi.integrate_scatter(EvaluationFlags::values, dst);
          phi_neighbor.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    apply_boundary(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, dim + 1, Number> phi(data, true, 0, 0, 0);
      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          phi.reinit(face);
          phi.gather_evaluate(src, EvaluationFlags::values);

          const VectorizedArray<Number> rho     = phi.read_cell_data(densities);
          const VectorizedArray<Number> rho_inv = 1. / rho;
          const VectorizedArray<Number> c       = phi.read_cell_data(speeds);
          const VectorizedArray<Number> c_sq_rho = c * c * rho;
          const VectorizedArray<Number> tau      = 1. / (c * rho);

          const int boundary_id = int(data.get_boundary_id(face));

          for (const unsigned int q : phi.quadrature_point_indices())
            {
              Tensor<1, dim, VectorizedArray<Number>> normal =
                phi.get_normal_vector(q);
              Tensor<1, dim + 1, VectorizedArray<Number>> val_plus =
                phi.get_value(q);
              VectorizedArray<Number> p_plus        = val_plus[dim];
              VectorizedArray<Number> normal_v_plus = val_plus[0] * normal[0];
              for (unsigned int d = 1; d < dim; ++d)
                normal_v_plus += val_plus[d] * normal[d];

              VectorizedArray<Number> lambda;
              switch (boundary_id)
                {
                  case 1: // soft wall - normal velocity component is zero
                    {
                      lambda = 1. / tau * normal_v_plus + p_plus;
                      break;
                    }
                  case 2: // hard wall - pressure is zero
                    {
                      lambda = VectorizedArray<Number>();
                      break;
                    }
                  case 3: // absorbing wall - mimics an open domain by the first
                          // order absorbing condition
                    {
                      lambda = 0.5 * p_plus + 0.5 / tau * normal_v_plus;
                      break;
                    }
                  default:
                    Assert(
                      false,
                      ExcMessage(
                        "set your boundary ids correctly: 1 - soft wall, 2 - hard wall, 3 - first order ABC"));
                }

              for (unsigned int d = 0; d < dim; ++d)
                val_plus[d] = (lambda - p_plus) * rho_inv * normal[d];
              val_plus[dim] =
                c_sq_rho * (normal_v_plus - tau * (lambda - p_plus));

              phi.submit_value(val_plus, q);
            }
          phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
  };



  template <int dim, typename Number>
  class InverseMatrixOperator
  {
  public:
    InverseMatrixOperator(const MatrixWaveOperation<dim, Number> &op)
      : op(op){};

    void
    vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      op.apply_inverse_mass_matrix(dst, src);
    }

  private:
    const MatrixWaveOperation<dim, Number> &op;
  };



  template <int dim, typename Number>
  class DiagonallyImplicitRungeKuttaIntegrator
  {
  public:
    DiagonallyImplicitRungeKuttaIntegrator(
      const unsigned int                      n_stages,
      const MatrixWaveOperation<dim, Number> &op)
      : op(const_cast<MatrixWaveOperation<dim, Number> &>(op))
      , n_accumulated_iterations(0)
      , n_solutions(0)
    {
      b.resize(n_stages);
      A.reinit(n_stages, n_stages);
      if (n_stages == 1)
        {
          // backward Euler
          b[0]    = 1.0;
          A(0, 0) = 1.;
        }
      else if (n_stages == 3)
        {
          // third order, stiffly stable, by Alexander (or p 77/formula (229)
          // of Kennedy & Carpenter, 2016)
          const double gamma = 0.4358665215084589994160194;
          const double alpha = 1 + gamma * (-4 + 2 * gamma);
          const double beta  = -1 + gamma * (6 + gamma * (-9 + 3 * gamma));
          b                  = {{(-1 + 4 * gamma) / (4 * beta),
                                 -0.75 * alpha * alpha / beta,
                                 gamma}};
          for (unsigned int d = 0; d < 3; ++d)
            A(d, d) = gamma;
          const double c2 = (2 + gamma * (-9 + 6 * gamma)) / (3 * alpha);
          A(1, 0)         = c2 - gamma;
          for (unsigned int d = 0; d < 2; ++d)
            A(2, d) = b[d];
        }
      else if (n_stages == 4)
        {
          // third order, stiffly stable, by formula (237) on page 82 of
          // Kennedy & Carpenter, 2016
          const double gamma = 9. / 40.;
          b = {{4032. / 9943., 6929. / 15485., -723. / 9272., gamma}};
          for (unsigned int d = 0; d < 4; ++d)
            A(d, d) = gamma;
          A(1, 0) = 163. / 520.;
          A(2, 0) = -6481433. / 8838675.;
          A(2, 1) = 87795409. / 70709400.;
          for (unsigned int d = 0; d < 3; ++d)
            A(3, d) = b[d];
        }
    }

    void
    perform_time_step(LinearAlgebra::distributed::Vector<Number> &solution,
                      LinearAlgebra::distributed::Vector<Number> &tmp,
                      const double                                time_step)
    {
      std::vector<LinearAlgebra::distributed::Vector<Number>> ki(b.size());
      LinearAlgebra::distributed::Vector<Number>              tmp2;
      tmp2               = solution;
      const double gamma = A(0, 0);
      for (unsigned int stage = 0; stage < b.size(); ++stage)
        {
          ki[stage].reinit(solution);
          op.set_time_step(0.);
          // note that we store what is called ki in Runge-Kutta methods as ki
          // / (gamma * dt) for simpler manipulation
          for (unsigned int r = 1; r < stage; ++r)
            tmp2.add((A(stage - 1, r - 1) - A(stage, r - 1)) / gamma,
                     ki[r - 1]);
          if (stage > 0)
            tmp2.add(-A(stage, stage - 1) / gamma, ki[stage - 1]);
          op.vmult(tmp, tmp2);
          op.set_time_step(gamma * time_step);

          SolverControl control(1000, tmp.l2_norm() * 1e-8);
          SolverGMRES<LinearAlgebra::distributed::Vector<Number>> solver(
            control);
          solver.solve(op,
                       ki[stage],
                       tmp,
                       InverseMatrixOperator<dim, Number>(op));
          n_accumulated_iterations += control.last_step();
          ++n_solutions;
          solution.add(-b[stage] / gamma, ki[stage]);
        }
    }

    std::pair<std::size_t, std::size_t>
    get_solver_statistics() const
    {
      return std::make_pair(n_accumulated_iterations, n_solutions);
    }

  private:
    MatrixWaveOperation<dim, Number> &op;
    std::vector<double>               b;
    FullMatrix<double>                A;

    std::size_t n_accumulated_iterations;
    std::size_t n_solutions;
  };



  template <int dim>
  class MyTriangulation :
#ifdef DEAL_II_WITH_P4EST
    public parallel::distributed::Triangulation<dim>
#else
    public Triangulation<dim>
#endif
  {
  public:
    MyTriangulation(const MPI_Comm communicator)
#ifdef DEAL_II_WITH_P4EST
      : parallel::distributed::Triangulation<dim>(communicator)
#endif
    {
      (void)communicator;
    }
  };



  template <>
  class MyTriangulation<1> : public Triangulation<1>
  {
  public:
    MyTriangulation(const MPI_Comm)
    {}

    MPI_Comm
    get_mpi_communicator() const override
    {
      return MPI_COMM_SELF;
    }
  };



  // Class WaveEquationProblem as  base class for this setup. It holds all
  // necessary informations like triangulation, dof handler, ...
  template <int dim>
  class WaveEquationProblem
  {
  public:
    typedef typename WaveEquationOperationBase<dim>::value_type value_type;
    WaveEquationProblem(Parameters &parameters_in);
    void
    run();
    bool
    cfl_stable()
    {
      return !(last_error_val > 100.0 * first_error_val ||
               last_error_val > 1.5 * first_mangnitude_val);
    }

  private:
    void
    make_grid();
    void
    make_dofs();

    void
    output_results();

    void
    adapt_mesh();

    LinearAlgebra::distributed::Vector<value_type> solutions, tmp_solutions;
    LinearAlgebra::distributed::Vector<value_type> post_pressure;

    TimeControl time_control;

    ConditionalOStream pcout;

    Parameters               &parameters;
    MyTriangulation<dim>      triangulation;
    MappingQGeneric<dim>      mapping;
    FESystem<dim>             fe;
    FE_DGQArbitraryNodes<dim> fe_spectral, fe_post_disp;
    DoFHandler<dim> dof_handler, dof_handler_spectral, dof_handler_post_disp;
    IndexSet        locally_relevant_dofs, loc_disp;
    std::shared_ptr<WaveEquationOperationBase<dim>> wave_equation_op;
    double                                          maximal_cellwise_error_init;

    // help variables for cfl stability anlysis
    double last_error_val;
    double first_error_val;
    double first_mangnitude_val;
  };



  template <int dim>
  WaveEquationProblem<dim>::WaveEquationProblem(Parameters &parameters_in)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , parameters(parameters_in)
    , triangulation(MPI_COMM_WORLD)
    , mapping(parameters.fe_degree)
    , fe(FE_DGQ<dim>(parameters.fe_degree), dim + 1)
    ,
    // fe(FE_DGQArbitraryNodes<dim>(QGauss<1>(fe_degree+1)),dim+1),
    fe_spectral(QGauss<1>(parameters.fe_degree + 1))
    , fe_post_disp(QGaussLobatto<1>(parameters.fe_degree + 2))
    , dof_handler(triangulation)
    , dof_handler_spectral(triangulation)
    , dof_handler_post_disp(triangulation)
    , maximal_cellwise_error_init(-1)
    , first_error_val(-1.0)
  {}



  template <int dim>
  void
  WaveEquationProblem<dim>::make_grid()
  {
    input_geometry_description(triangulation, parameters);

    pcout << "Number of global active cells: "
          << triangulation.n_global_active_cells() << std::endl;

    {
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);
      Utilities::MPI::MinMaxAvg memory =
        Utilities::MPI::min_max_avg(stats.VmRSS / 1024,
                                    triangulation.get_mpi_communicator());
      pcout << "   Memory stats [MB]: " << memory.min << " " << memory.avg
            << " " << memory.max << std::endl;
    }
  }



  template <int dim>
  void
  WaveEquationProblem<dim>::make_dofs()
  {
    Timer time;
    dof_handler.distribute_dofs(fe);
    dof_handler_spectral.distribute_dofs(fe_spectral);
    time.restart();
    dof_handler_post_disp.distribute_dofs(fe_post_disp);

    time.restart();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "Number of degrees of freedom DG system: " << dof_handler.n_dofs()
          << std::endl;

    // Add second DoFHandler object for the fast computation of the
    // post-processing
    std::vector<const DoFHandler<dim> *> dof_handlers(3);
    dof_handlers[0] = &dof_handler;
    dof_handlers[2] = &dof_handler_spectral;
    dof_handlers[1] = &dof_handler_post_disp;

    time_control.set_time_step(
      compute_time_step_size(triangulation, parameters));
    wave_equation_op->setup(mapping, dof_handlers, input_materials());

    time.restart();
    wave_equation_op->get_matrix_free().initialize_dof_vector(solutions);
    tmp_solutions.reinit(solutions);
    wave_equation_op->get_matrix_free().initialize_dof_vector(post_pressure, 1);

    {
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);
      Utilities::MPI::MinMaxAvg memory =
        Utilities::MPI::min_max_avg(stats.VmRSS / 1024,
                                    triangulation.get_mpi_communicator());
      pcout << "   Memory stats [MB]: " << memory.min << " " << memory.avg
            << " " << memory.max << std::endl;
    }
    pcout << "   Time vectors: " << time.wall_time() << std::endl;
  }



  template <int dim>
  void
  set_refinement_indicators(parallel::distributed::Triangulation<dim> &tria,
                            const Vector<double> &error_per_cell)
  {
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      tria, error_per_cell, 0.1, 0.6);
  }



  template <int dim>
  void
  set_refinement_indicators(Triangulation<dim>   &tria,
                            const Vector<double> &error_per_cell)
  {
    GridRefinement::refine_and_coarsen_fixed_number(tria,
                                                    error_per_cell,
                                                    0.1,
                                                    0.6);
  }



  template <int dim>
  void
  WaveEquationProblem<dim>::adapt_mesh()
  {
    Vector<double> error_per_cell(triangulation.n_active_cells());
    wave_equation_op->estimate_error(solutions, tmp_solutions, error_per_cell);

    const int min_level = parameters.n_refinements;
    const int max_level = min_level + parameters.n_adaptive_refinements;

    set_refinement_indicators(triangulation, error_per_cell);

    // In order to avoid refining too much (waves tend to scatter and occupy
    // the whole domain), we try to coarsen as soon as the error estimate
    // becomes small as compared to the error in the initial condition. The
    // idea is that the initial condition can guide as an order of magnitude
    // for the largest error components that appear during a simulation.
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
         cell != triangulation.end();
         ++cell)
      if (cell->is_locally_owned())
        {
          if (cell->refine_flag_set() && cell->level() == max_level)
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() && cell->level() == min_level)
            cell->clear_coarsen_flag();
          if (cell->refine_flag_set() &&
              error_per_cell(cell->active_cell_index()) <
                0.1 * maximal_cellwise_error_init)
            cell->clear_refine_flag();
          if (error_per_cell(cell->active_cell_index()) <
              0.05 * maximal_cellwise_error_init)
            cell->set_coarsen_flag();
        }


#ifdef DEAL_II_WITH_P4EST
    // parallel::distributed::SolutionTransfer does not exist for 1D, so make
    // sure we only use valid objects. Obviously, this code is going to fail
    // in 1D, so we have an AssertThrow that makes sure this is only executed
    // in higher dimensions
    AssertThrow(dim > 1, ExcNotImplemented());

    parallel::distributed::SolutionTransfer<
      (dim > 1 ? dim : 2),
      LinearAlgebra::distributed::Vector<value_type>>
      sol_trans(*reinterpret_cast<const DoFHandler<(dim > 1 ? dim : 2)> *>(
        &dof_handler));
    triangulation.prepare_coarsening_and_refinement();
    sol_trans.prepare_for_coarsening_and_refinement(solutions);
    triangulation.execute_coarsening_and_refinement();
    make_dofs();
    sol_trans.interpolate(solutions);
#else
    SolutionTransfer<dim, LinearAlgebra::distributed::Vector<value_type>>
      sol_trans(dof_handler);
    triangulation.prepare_coarsening_and_refinement();
    LinearAlgebra::distributed::Vector<value_type> xsol(solutions);
    sol_trans.prepare_for_coarsening_and_refinement(xsol);
    triangulation.execute_coarsening_and_refinement();
    make_dofs();
    sol_trans.interpolate(xsol, solutions);
#endif

    // ader_lts sets the time step size in its setup routine (called in
    // make_dofs) the other integrators do not update the time themselves and
    // have to get it from the compute_time_step_size routine
    if (parameters.integ_type != IntegratorType::ader_lts)
      time_control.set_time_step(
        compute_time_step_size(triangulation, parameters));
  }



  template <int dim>
  void
  WaveEquationProblem<dim>::output_results()
  {
    if (parameters.write_vtu_output)
      {
        Vector<double> procs(triangulation.n_active_cells()),
          clusterids(triangulation.n_active_cells()),
          timestepsizes(triangulation.n_active_cells()),
          is(triangulation.n_active_cells()),
          vs(triangulation.n_active_cells());
        for (unsigned int i = 0;
             i < wave_equation_op->get_matrix_free().n_cell_batches();
             ++i)
          for (unsigned int v = 0; v < wave_equation_op->get_matrix_free()
                                         .n_active_entries_per_cell_batch(i);
               ++v)
            {
              typename Triangulation<dim>::cell_iterator cell =
                wave_equation_op->get_matrix_free().get_cell_iterator(i, v);
              procs(cell->active_cell_index()) =
                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
              clusterids(cell->active_cell_index()) =
                wave_equation_op->cluster_id(i) + 1.0;
              timestepsizes(cell->active_cell_index()) =
                wave_equation_op->time_step(i);
              is(cell->active_cell_index()) = i;
              vs(cell->active_cell_index()) = v;
            }

        DataOut<dim> data_out;

        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);

        data_out.attach_dof_handler(dof_handler);
        std::vector<std::string> solution_names;
        for (unsigned int d = 0; d < dim; ++d)
          solution_names.push_back("solution_velocity");
        solution_names.push_back("solution_pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
          DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(dof_handler,
                                 solutions,
                                 solution_names,
                                 interpretation);
        LinearAlgebra::distributed::Vector<value_type> vec(solutions);
        wave_equation_op->project_initial_field(
          vec,
          ExactSolution<dim>(dim + 1,
                             -1,
                             time_control.get_time(),
                             parameters.initial_cases,
                             parameters.membrane_modes));
        vec -= solutions;
        for (unsigned int d = 0; d < dim; ++d)
          solution_names[d] = "error_velocity";
        solution_names[dim] = "error_pressure";
        data_out.add_data_vector(dof_handler,
                                 vec,
                                 solution_names,
                                 interpretation);
        Vector<double> error_estimate(triangulation.n_active_cells());
        wave_equation_op->estimate_error(solutions,
                                         tmp_solutions,
                                         error_estimate);
        data_out.add_data_vector(error_estimate, "Error_estimate");
        if (parameters.integ_type == IntegratorType::ader_lts)
          {
            data_out.add_data_vector(clusterids, "cluster_id");
            data_out.add_data_vector(timestepsizes, "time_step");
          }
#ifdef DEBUG
        data_out.add_data_vector(procs, "MPI_Proc_id");
        data_out.add_data_vector(is, "macrocell_i_index");
        data_out.add_data_vector(vs, "macrocell_v_index");
#endif
        data_out.add_data_vector(dof_handler_post_disp,
                                 post_pressure,
                                 "post_pressure");
        data_out.build_patches(mapping,
                               parameters.fe_degree,
                               DataOut<dim>::curved_inner_cells);

        const std::string filename_pressure =
          "sol_deg" + Utilities::int_to_string(parameters.fe_degree, 1) + "_" +
          wave_equation_op->Name() + "_case" +
          Utilities::int_to_string(parameters.initial_cases, 1) + "_ref" +
          Utilities::int_to_string(parameters.n_refinements, 1) + "_step" +
          Utilities::int_to_string(time_control.get_output_step_number(), 3);

        {
          std::ostringstream filename;
          filename << "output/" << filename_pressure;
          if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
            filename << "_Proc"
                     << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
          filename << ".vtu";

          std::ofstream output_pressure(filename.str().c_str());
          data_out.write_vtu(output_pressure);
        }


        if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1 &&
            Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::vector<std::string> filenames;
            for (unsigned int i = 0;
                 i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
                 ++i)
              {
                std::ostringstream filename;
                filename << filename_pressure << "_Proc" << i << ".vtu";

                filenames.push_back(filename.str().c_str());
              }
            std::string   master_name = "output/" + filename_pressure + ".pvtu";
            std::ofstream master_output(master_name.c_str());
            data_out.write_pvtu_record(master_output, filenames);
          }
      }


    Vector<double> norm_per_cell_p(triangulation.n_active_cells());

    ComponentSelectFunction<dim> pressure_select(dim, dim + 1);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solutions,
                                      Functions::ZeroFunction<dim>(dim + 1),
                                      norm_per_cell_p,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::L2_norm,
                                      &pressure_select);
    double solution_mag = std::sqrt(
      Utilities::MPI::sum(norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));
    double solution_norm_p = 0.0, solution_norm_v = 0.0,
           solution_norm_p_post = 0.0;

    VectorTools::integrate_difference(
      mapping,
      dof_handler,
      solutions,
      ExactSolution<dim>(dim + 1,
                         dim,
                         time_control.get_time(),
                         parameters.initial_cases,
                         parameters.membrane_modes),
      norm_per_cell_p,
      QGauss<dim>(fe.degree + 2),
      VectorTools::L2_norm,
      &pressure_select);

    last_error_val = solution_norm_p = std::sqrt(
      Utilities::MPI::sum(norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    ComponentSelectFunction<dim> velocity_select(
      std::pair<unsigned int, unsigned int>(0U, dim), dim + 1);
    VectorTools::integrate_difference(
      mapping,
      dof_handler,
      solutions,
      ExactSolution<dim>(dim + 1,
                         -1,
                         time_control.get_time(),
                         parameters.initial_cases,
                         parameters.membrane_modes),
      norm_per_cell_p,
      QGauss<dim>(fe.degree + 2),
      VectorTools::L2_norm,
      &velocity_select);
    solution_norm_v = std::sqrt(
      Utilities::MPI::sum(norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    // compute post pressure
    wave_equation_op->compute_post_pressure(solutions,
                                            tmp_solutions,
                                            post_pressure);
    VectorTools::integrate_difference(
      mapping,
      dof_handler_post_disp,
      post_pressure,
      ExactSolution<dim>(1,
                         dim,
                         time_control.get_time(),
                         parameters.initial_cases,
                         parameters.membrane_modes),
      norm_per_cell_p,
      QGauss<dim>(fe.degree + 3),
      VectorTools::L2_norm);

    solution_norm_p_post = std::sqrt(
      Utilities::MPI::sum(norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    tmp_solutions = 0;
    VectorTools::integrate_difference(
      mapping,
      dof_handler,
      tmp_solutions,
      ExactSolution<dim>(dim + 1,
                         dim,
                         time_control.get_time(),
                         parameters.initial_cases,
                         parameters.membrane_modes),
      norm_per_cell_p,
      QGauss<dim>(fe.degree + 2),
      VectorTools::L2_norm,
      &pressure_select);
    solution_mag = std::sqrt(
      Utilities::MPI::sum(norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    if (parameters.cfl_stability_analysis)
      {
        if (first_error_val < 0.0)
          {
            first_error_val      = last_error_val;
            first_mangnitude_val = solution_mag;
          }
        if (last_error_val > 100.0 * first_error_val ||
            last_error_val > 1.5 * first_mangnitude_val)
          time_control.set_time(parameters.final_time);
      }

    pcout << "   Time:" << std::fixed << std::setw(8) << std::setprecision(2)
          << time_control.get_time() << " , error p: " << std::scientific
          << std::setprecision(4) << std::setw(10) << solution_norm_p
          << " , error p post: " << std::scientific << std::setprecision(4)
          << std::setw(10) << solution_norm_p_post
          << " , error v: " << std::scientific << std::setprecision(4)
          << std::setw(10) << solution_norm_v
          << " , solution mag p: " << std::scientific << std::setprecision(4)
          << std::setw(10) << solution_mag << std::endl;

    if (false)
      pcout << "write output for time step " << time_control.get_step_number()
            << " at time " << std::fixed << std::setprecision(2)
            << time_control.get_time() << std::endl;
  }



  template <int dim>
  double
  compute_time_step_size(const Triangulation<dim> &triangulation,
                         Parameters               &parameters)
  {
    typename Triangulation<dim>::active_cell_iterator cell = triangulation
                                                               .begin_active(),
                                                      endc =
                                                        triangulation.end();
    double min_cell_diameter = std::numeric_limits<double>::max();
    double diameter          = 0.0;

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          diameter = cell->minimum_vertex_distance() /
                     input_materials()[cell->material_id()].speed;
          if (diameter < min_cell_diameter)
            min_cell_diameter = diameter;
        }

    return parameters.cfl_number *
           Utilities::MPI::min(min_cell_diameter, MPI_COMM_WORLD);
  }



  template <int dim>
  void
  WaveEquationProblem<dim>::run()
  {
    using VectorType = LinearAlgebra::distributed::Vector<value_type>;

    make_grid();

    // setup time control
    time_control.setup(parameters.final_time,
                       parameters.output_every_time,
                       compute_time_step_size(triangulation, parameters),
                       parameters.max_time_steps);

    pcout << "Time step size: " << time_control.get_time_step() << std::endl
          << std::endl;

    // determine wave equation operation, i.e. how to evaluate the integrals
    switch (parameters.integ_type)
      {
        // implicit time integration -> do nothing
        case IntegratorType::impleuler:
        case IntegratorType::expleuler:
        case IntegratorType::classrk4:
        case IntegratorType::lsrk45reg2:
        case IntegratorType::lsrk33reg2:
        case IntegratorType::lsrk45reg3:
        case IntegratorType::lsrk59reg2:
        case IntegratorType::ssprk:
          {
            if (parameters.fe_degree == 1)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 1>(time_control, parameters));
            else if (parameters.fe_degree == 2)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 2>(time_control, parameters));
            else if (parameters.fe_degree == 3)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 3>(time_control, parameters));
            else if (parameters.fe_degree == 4)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 4>(time_control, parameters));
            else if (parameters.fe_degree == 5)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 5>(time_control, parameters));
            else if (parameters.fe_degree == 6)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 6>(time_control, parameters));
            else if (parameters.fe_degree == 7)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 7>(time_control, parameters));
            else if (parameters.fe_degree == 8)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 8>(time_control, parameters));
            else if (parameters.fe_degree == 9)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 9>(time_control, parameters));
            else if (parameters.fe_degree == 10)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 10>(time_control, parameters));
            else if (parameters.fe_degree == 11)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 11>(time_control, parameters));
            else if (parameters.fe_degree == 12)
              wave_equation_op.reset(
                new WaveEquationOperation<dim, 12>(time_control, parameters));
            else
              Assert(false, ExcNotImplemented());
            break;
          }
        case IntegratorType::ader:
          {
            if (parameters.fe_degree == 1)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 1>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 2)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 2>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 3)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 3>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 4)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 4>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 5)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 5>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 6)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 6>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 7)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 7>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 8)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 8>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 9)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 9>(time_control,
                                                      parameters));
            else if (parameters.fe_degree == 10)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 10>(time_control,
                                                       parameters));
            else if (parameters.fe_degree == 11)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 11>(time_control,
                                                       parameters));
            else if (parameters.fe_degree == 12)
              wave_equation_op.reset(
                new WaveEquationOperationADER<dim, 12>(time_control,
                                                       parameters));
            else
              Assert(false, ExcNotImplemented());
            break;
          }
        case IntegratorType::ader_lts:
          {
            // after this call, the variable time_step is set to the biggest
            // time_step of the LTS scheme
            if (parameters.fe_degree == 1)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 1>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 2)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 2>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 3)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 3>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 4)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 4>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 5)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 5>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 6)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 6>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 7)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 7>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 8)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 8>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 9)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 9>(time_control,
                                                         parameters));
            else if (parameters.fe_degree == 10)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 10>(time_control,
                                                          parameters));
            else if (parameters.fe_degree == 11)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 11>(time_control,
                                                          parameters));
            else if (parameters.fe_degree == 12)
              wave_equation_op.reset(
                new WaveEquationOperationADERLTS<dim, 12>(time_control,
                                                          parameters));
            else
              Assert(false, ExcNotImplemented());
            break;
          }
        case IntegratorType::ader_adconfull:
          {
            if (parameters.fe_degree == 1)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 1>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 2)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 2>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 3)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 3>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 4)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 4>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 5)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 5>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 6)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 6>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 7)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 7>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 8)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 8>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 9)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 9>(time_control,
                                                               parameters));
            else if (parameters.fe_degree == 10)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 10>(time_control,
                                                                parameters));
            else if (parameters.fe_degree == 11)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 11>(time_control,
                                                                parameters));
            else if (parameters.fe_degree == 12)
              wave_equation_op.reset(
                new WaveEquationOperationADERADCONFULL<dim, 12>(time_control,
                                                                parameters));
            else
              Assert(false, ExcNotImplemented());
            break;
          }
        default:
          Assert(false, ExcNotImplemented());
      }

    make_dofs();
    pcout << "   Time step size: " << time_control.get_time_step() << std::endl;


    // set initial conditions
    wave_equation_op->project_initial_field(
      solutions,
      ExactSolution<dim>(dim + 1,
                         -1,
                         time_control.get_time(),
                         parameters.initial_cases,
                         parameters.membrane_modes));


    unsigned int n_refinements_left = parameters.n_adaptive_refinements;
    while (n_refinements_left > 0)
      {
        adapt_mesh();
        wave_equation_op->project_initial_field(
          solutions,
          ExactSolution<dim>(dim + 1,
                             -1,
                             time_control.get_time(),
                             parameters.initial_cases,
                             parameters.membrane_modes));
        --n_refinements_left;
        if (n_refinements_left == 0)
          {
            Vector<double> error_per_cell(triangulation.n_active_cells());
            wave_equation_op->estimate_error(solutions,
                                             tmp_solutions,
                                             error_per_cell);
            maximal_cellwise_error_init =
              Utilities::MPI::max(error_per_cell.linfty_norm(), MPI_COMM_WORLD);
          }
      }

    MatrixWaveOperation<dim, value_type> matrix(wave_equation_op->get_matrix_free());
    matrix.reinit(input_materials());
    matrix.set_time_step(time_control.get_time_step());

    // output initial fields
    output_results();

    // determine integrator, i.e. how to combine the state vectors
    std::shared_ptr<
      ExplicitIntegrator<VectorType, WaveEquationOperationBase<dim>>>
      integrator;
    switch (parameters.integ_type)
      {
        case IntegratorType::expleuler:
          {
            integrator.reset(
              new ExplicitEuler<VectorType, WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::classrk4:
          {
            integrator.reset(
              new ClassRK4<VectorType, WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::lsrk45reg2:
          {
            integrator.reset(
              new LowStorageRK45Reg2<VectorType,
                                     WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::lsrk33reg2:
          {
            integrator.reset(
              new LowStorageRK33Reg2<VectorType,
                                     WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::lsrk45reg3:
          {
            integrator.reset(
              new LowStorageRK45Reg3<VectorType,
                                     WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::lsrk59reg2:
          {
            integrator.reset(
              new LowStorageRK59Reg2<VectorType,
                                     WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::ssprk:
          {
            integrator.reset(
              new SSPRK<VectorType, WaveEquationOperationBase<dim>>(4, 8));
            break;
          }
        case IntegratorType::ader:
        case IntegratorType::ader_adconfull:
          {
            integrator.reset(
              new ArbitraryHighOrderDG<VectorType,
                                       WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::ader_lts:
          {
            integrator.reset(
              new ArbitraryHighOrderDGLTS<VectorType,
                                          WaveEquationOperationBase<dim>>());
            break;
          }
        case IntegratorType::impleuler:
          break;

        default:
          Assert(false, ExcNotImplemented());
      }

    DiagonallyImplicitRungeKuttaIntegrator<dim, value_type> implicit_integrator(
      4, matrix);

    Timer  timer;
    double wtime       = 0.0;
    double output_time = 0.0;
    while (!time_control.done())
      {
        time_control.advance_time_step();

        timer.restart();

        if (parameters.integ_type == IntegratorType::impleuler)
          {
            implicit_integrator.perform_time_step(solutions,
                                                  tmp_solutions,
                                                  time_control.get_time_step());
          }
        else
          {
            tmp_solutions.swap(solutions);
            integrator->perform_time_step(tmp_solutions,
                                          solutions,
                                          time_control.get_time_step(),
                                          *wave_equation_op);
          }
        wtime += timer.wall_time();

        if (parameters.n_adaptive_refinements > 0)
          if (time_control.get_step_number() %
                parameters.adaptive_refinement_interval ==
              0)
            adapt_mesh();

        timer.restart();
        time_step_analysis(mapping,
                           dof_handler,
                           solutions,
                           time_control.get_time());

        if (time_control.at_tick())
          output_results();
        output_time += timer.wall_time();
      }

    pcout << std::endl
          << "   Performed " << time_control.get_step_number() << " time steps."
          << std::endl;
    if (parameters.integ_type == IntegratorType::impleuler)
      {
        pcout << "   Statistics of linear solver: n_systems = "
              << implicit_integrator.get_solver_statistics().second
              << ", avg_its = "
              << implicit_integrator.get_solver_statistics().first /
                   implicit_integrator.get_solver_statistics().second
              << std::endl;
      }

    pcout << "   Average wallclock time per time step: "
          << wtime / time_control.get_step_number() << "s, time per element: "
          << wtime / time_control.get_step_number() /
               triangulation.n_active_cells()
          << "s" << std::endl;

    pcout << "   Spent " << output_time << " s on output";
    pcout << "   and   " << Utilities::MPI::max(wtime, MPI_COMM_WORLD)
          << " s on computations." << std::endl;
  }

  void
  run_cfl_stability_analysis(Parameters &parameters_in)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    double cfl_test             = parameters_in.cfl_number;
    double cfl_closest_stable   = -0.1;
    double cfl_closest_instable = 100.0;

    for (unsigned int i = 0; i < 12; ++i)
      {
        pcout << "*******************************************************"
              << std::endl;
        pcout << "cfl " << cfl_test * std::pow(parameters_in.fe_degree, 1.5)
              << " in iteration " << i << std::endl;
        pcout << "*******************************************************"
              << std::endl;
        pcout << std::endl;


        // run the problem in 2D or 3D and record error parameters
        bool simu_was_stable = false;
        if (parameters_in.dimension == 2)
          {
            WaveEquationProblem<2> we_problem(parameters_in);
            we_problem.run();
            simu_was_stable = we_problem.cfl_stable();
          }
        else if (parameters_in.dimension == 3)
          {
            WaveEquationProblem<3> we_problem(parameters_in);
            we_problem.run();
            simu_was_stable = we_problem.cfl_stable();
          }

        if (simu_was_stable)
          cfl_closest_stable = cfl_test;
        else
          cfl_closest_instable = cfl_test;

        if (cfl_closest_stable < 0.0)
          {
            if (cfl_test / std::pow(parameters_in.fe_degree, 1.5) > 0.15)
              cfl_test -= 0.1;
            else
              cfl_test /= 3.0;
          }
        else if (cfl_closest_instable > 99.0)
          cfl_test += 0.05;
        else
          cfl_test = (cfl_closest_instable + cfl_closest_stable) / 2.0;

        parameters_in.cfl_number = cfl_test;
      }

    pcout << "*******************************************************"
          << std::endl;
    pcout << "Final results for the CFL stability analysis:" << std::endl;
    pcout << "The Courant number                "
          << cfl_closest_instable * std::pow(parameters_in.fe_degree, 1.5)
          << " is instable" << std::endl;
    pcout << "The Courant number                "
          << cfl_closest_stable * std::pow(parameters_in.fe_degree, 1.5)
          << " is stable" << std::endl;
    pcout << "The limit might be in the middle: "
          << (cfl_closest_instable + cfl_closest_instable) *
               std::pow(parameters_in.fe_degree, 1.5) / 2.0
          << std::endl;
    pcout << "*******************************************************"
          << std::endl;
    pcout << std::endl;

    return;
  }
} // namespace HDG_WE

int
main(int argc, char **argv)
{
  using namespace HDG_WE;
  using namespace dealii;

#ifdef __x86_64

  // on x86-64:
  // change mode for rounding: denormals are flushed to zero to avoid computing
  // on denormals which can slow down computations a lot.
#  define MXCSR_DAZ (1 << 6)  /* Enable denormals are zero mode */
#  define MXCSR_FTZ (1 << 15) /* Enable flush to zero mode */

  unsigned int mxcsr = __builtin_ia32_stmxcsr();
  mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
  __builtin_ia32_ldmxcsr(mxcsr);

#endif

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << std::endl
                    << "deal.II git version " << DEAL_II_GIT_SHORTREV
                    << " on branch " << DEAL_II_GIT_BRANCH << std::endl;
          std::cout << "Number of MPI ranks:         "
                    << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                    << std::endl;

          const unsigned int n_vect_doubles = VectorizedArray<double>::size();
          const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

          std::cout << "Vectorization over " << n_vect_doubles
                    << " doubles = " << n_vect_bits << " bits (";

          if (n_vect_bits == 64)
            std::cout << "disabled";
          else if (n_vect_bits == 128)
            std::cout << "SSE2";
          else if (n_vect_bits == 256)
            std::cout << "AVX";
          else if (n_vect_bits == 512)
            std::cout << "AVX512";
          else
            std::cout << "unknown";

          std::cout << "), VECTORIZATION_LEVEL="
                    << DEAL_II_COMPILER_VECTORIZATION_LEVEL << std::endl
                    << std::endl;
        }

      deallog.depth_console(0);

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "default_parameters.prm";
      Parameters parameters;
      parameters.read_parameters(paramfile);

      if (parameters.dimension == 2)
        {
          if (!parameters.cfl_stability_analysis)
            {
              WaveEquationProblem<2> we_problem(parameters);
              we_problem.run();
            }
          else
            run_cfl_stability_analysis(parameters);
        }
      else if (parameters.dimension == 3)
        {
          if (!parameters.cfl_stability_analysis)
            {
              WaveEquationProblem<3> we_problem(parameters);
              we_problem.run();
            }
          else
            run_cfl_stability_analysis(parameters);
        }
      else
        AssertThrow(false,
                    ExcMessage("Invalid dimension " +
                               std::to_string(parameters.dimension)));


      // output of the used parameters to be able to rerun the simulation
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        parameters.output_parameters(std::cout);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
