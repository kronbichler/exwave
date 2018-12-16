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


#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/revision.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#include "../include/input_parameters.h"
#include "../include/parameters.h"
#include "../include/time_integrators.h"
#include "../include/wave_equation_operations.h"

namespace HDG_WE
{
  using namespace dealii;



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
      :
      parallel::distributed::Triangulation<dim>(communicator)
#endif
    {
      (void)communicator;
    }

#ifndef DEAL_II_WITH_P4EST
    MPI_Comm get_communicator() const
    {
      return MPI_COMM_SELF;
    }
#endif
  };



  template <>
  class MyTriangulation <1> : public Triangulation<1>
  {
  public:
    MyTriangulation(const MPI_Comm)
    {}

    MPI_Comm get_communicator() const
    {
      return MPI_COMM_SELF;
    }
  };



  // Class WaveEquationProblem as  base class for this setup. It holds all
  // necessary informations like triangulation, dof handler, ...
  template<int dim>
  class WaveEquationProblem
  {
  public:
    typedef typename WaveEquationOperationBase<dim>::value_type value_type;
    WaveEquationProblem(Parameters &parameters_in);
    void run();
    bool cfl_stable()
    {
      return !(last_error_val>100.0*first_error_val || last_error_val>1.5*first_mangnitude_val);
    }
  private:
    void make_grid ();
    void make_dofs ();

    void output_results ();

    void adapt_mesh();

    LinearAlgebra::distributed::Vector<value_type> solutions, tmp_solutions;
    LinearAlgebra::distributed::Vector<value_type> post_pressure;

    TimeControl time_control;

    ConditionalOStream             pcout;

    Parameters                    &parameters;
    MyTriangulation<dim>           triangulation;
    MappingQGeneric<dim>           mapping;
    FESystem<dim>                  fe;
    FE_DGQArbitraryNodes<dim>      fe_spectral, fe_post_disp;
    DoFHandler<dim>                dof_handler, dof_handler_spectral, dof_handler_post_disp;
    IndexSet                       locally_relevant_dofs, loc_disp;
    std::shared_ptr<WaveEquationOperationBase<dim> > wave_equation_op;
    double                         maximal_cellwise_error_init;

    // help variables for cfl stability anlysis
    double last_error_val;
    double first_error_val;
    double first_mangnitude_val;
  };



  template<int dim>
  WaveEquationProblem<dim>::WaveEquationProblem(Parameters &parameters_in)
    :
    pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
    parameters(parameters_in),
    triangulation(MPI_COMM_WORLD),
    mapping(parameters.fe_degree),
    fe(FE_DGQ<dim>(parameters.fe_degree), dim+1),
    //fe(FE_DGQArbitraryNodes<dim>(QGauss<1>(fe_degree+1)),dim+1),
    fe_spectral(QGauss<1>(parameters.fe_degree+1)),
    fe_post_disp(QGaussLobatto<1>(parameters.fe_degree+2)),
    dof_handler(triangulation),
    dof_handler_spectral(triangulation),
    dof_handler_post_disp(triangulation),
    maximal_cellwise_error_init(-1),
    first_error_val(-1.0)
  {

  }



  template<int dim>
  void WaveEquationProblem<dim>::make_grid()
  {
    input_geometry_description(triangulation,parameters);

    pcout << "Number of global active cells: "
          << triangulation.n_global_active_cells()
          << std::endl;

    {
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);
      Utilities::MPI::MinMaxAvg memory =
        Utilities::MPI::min_max_avg (stats.VmRSS/1024, triangulation.get_communicator());
      pcout << "   Memory stats [MB]: " << memory.min << " "
            << memory.avg << " " << memory.max << std::endl;
    }

  }



  template<int dim>
  void WaveEquationProblem<dim>::make_dofs()
  {
    Timer time;
    dof_handler.distribute_dofs(fe);
    dof_handler_spectral.distribute_dofs(fe_spectral);
    time.restart();
    dof_handler_post_disp.distribute_dofs(fe_post_disp);

    time.restart();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "Number of degrees of freedom DG system: "
          << dof_handler.n_dofs()
          << std::endl;

    // Add second DoFHandler object for the fast computation of the
    // post-processing
    std::vector<const DoFHandler<dim> *> dof_handlers(3);
    dof_handlers[0] = &dof_handler;
    dof_handlers[2] = &dof_handler_spectral;
    dof_handlers[1] = &dof_handler_post_disp;

    time_control.set_time_step(compute_time_step_size(triangulation,parameters));
    wave_equation_op->setup(mapping,dof_handlers,input_materials());

    time.restart();
    wave_equation_op->get_matrix_free().initialize_dof_vector(solutions);
    tmp_solutions.reinit(solutions);
    wave_equation_op->get_matrix_free().initialize_dof_vector(post_pressure, 1);

    {
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);
      Utilities::MPI::MinMaxAvg memory =
        Utilities::MPI::min_max_avg (stats.VmRSS/1024, triangulation.get_communicator());
      pcout << "   Memory stats [MB]: " << memory.min << " "
            << memory.avg << " " << memory.max << std::endl;
    }
    pcout << "   Time vectors: " << time.wall_time() << std::endl;
  }



  template <int dim>
  void set_refinement_indicators(parallel::distributed::Triangulation<dim> &tria,
                                 const Vector<double> &error_per_cell)
  {
    parallel::distributed::GridRefinement::
    refine_and_coarsen_fixed_number(tria, error_per_cell,
                                    0.1, 0.6);
  }



  template <int dim>
  void set_refinement_indicators(Triangulation<dim> &tria,
                                 const Vector<double> &error_per_cell)
  {
    GridRefinement::
    refine_and_coarsen_fixed_number(tria, error_per_cell,
                                    0.1, 0.6);
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
           triangulation.begin_active(); cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned())
        {
          if (cell->refine_flag_set() && cell->level() == max_level)
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() && cell->level() == min_level)
            cell->clear_coarsen_flag();
          if (cell->refine_flag_set() &&
              error_per_cell(cell->active_cell_index())
              < 0.1 * maximal_cellwise_error_init)
            cell->clear_refine_flag();
          if (error_per_cell(cell->active_cell_index())
              < 0.05 * maximal_cellwise_error_init)
            cell->set_coarsen_flag();
        }


#ifdef DEAL_II_WITH_P4EST
    // parallel::distributed::SolutionTransfer does not exist for 1D, so make
    // sure we only use valid objects. Obviously, this code is going to fail
    // in 1D, so we have an AssertThrow that makes sure this is only executed
    // in higher dimensions
    AssertThrow(dim > 1, ExcNotImplemented());

    parallel::distributed::SolutionTransfer<(dim>1?dim:2),LinearAlgebra::distributed::Vector<value_type> >
    sol_trans(*reinterpret_cast<const DoFHandler<(dim>1?dim:2)>*>(&dof_handler));
    triangulation.prepare_coarsening_and_refinement();
    sol_trans.prepare_for_coarsening_and_refinement(solutions);
    triangulation.execute_coarsening_and_refinement ();
    make_dofs ();
    sol_trans.interpolate(solutions);
#else
    SolutionTransfer<dim,LinearAlgebra::distributed::Vector<value_type> >
    sol_trans(dof_handler);
    triangulation.prepare_coarsening_and_refinement();
    LinearAlgebra::distributed::Vector<value_type> xsol(solutions);
    sol_trans.prepare_for_coarsening_and_refinement(xsol);
    triangulation.execute_coarsening_and_refinement ();
    make_dofs ();
    sol_trans.interpolate(xsol, solutions);
#endif

    // ader_lts sets the time step size in its setup routine (called in make_dofs)
    // the other integrators do not update the time themselves and have to get it from
    // the compute_time_step_size routine
    if (parameters.integ_type!=IntegratorType::ader_lts)
      time_control.set_time_step(compute_time_step_size(triangulation,parameters));
  }



  template <int dim>
  void
  WaveEquationProblem<dim>::output_results ()
  {


    Vector<double> procs(triangulation.n_active_cells()),
           clusterids(triangulation.n_active_cells()),
           timestepsizes(triangulation.n_active_cells()),
           is(triangulation.n_active_cells()),
           vs(triangulation.n_active_cells());
    for (unsigned int i=0; i<wave_equation_op->get_matrix_free().n_macro_cells(); ++i)
      for (unsigned int v=0; v<wave_equation_op->get_matrix_free().n_components_filled(i); ++v)
        {
          typename Triangulation<dim>::cell_iterator cell = wave_equation_op->get_matrix_free().get_cell_iterator(i, v);
          procs(cell->active_cell_index()) = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
          clusterids(cell->active_cell_index()) = wave_equation_op->cluster_id(i)+1.0;
          timestepsizes(cell->active_cell_index()) = wave_equation_op->time_step(i);
          is(cell->active_cell_index()) = i;
          vs(cell->active_cell_index()) = v;
        }



    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler (dof_handler);
    std::vector<std::string> solution_names;
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("solution_velocity");
    solution_names.push_back("solution_pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector (dof_handler, solutions, solution_names, interpretation);
    LinearAlgebra::distributed::Vector<value_type> vec(solutions);
    wave_equation_op->project_initial_field(vec, ExactSolution<dim> (dim+1, -1, time_control.get_time(),parameters.initial_cases,parameters.membrane_modes));
    vec -= solutions;
    for (unsigned int d=0; d<dim; ++d)
      solution_names[d] = "error_velocity";
    solution_names[dim] = "error_pressure";
    data_out.add_data_vector (dof_handler, vec, solution_names, interpretation);
    Vector<double> error_estimate(triangulation.n_active_cells());
    wave_equation_op->estimate_error(solutions, tmp_solutions, error_estimate);
    data_out.add_data_vector (error_estimate, "Error_estimate");
    if (parameters.integ_type == IntegratorType::ader_lts)
      {
        data_out.add_data_vector (clusterids, "cluster_id");
        data_out.add_data_vector (timestepsizes, "time_step");
      }
#ifdef DEBUG
    data_out.add_data_vector (procs, "MPI_Proc_id");
    data_out.add_data_vector (is, "macrocell_i_index");
    data_out.add_data_vector (vs, "macrocell_v_index");
#endif
    data_out.add_data_vector (dof_handler_post_disp, post_pressure, "post_pressure");
    data_out.build_patches (mapping, parameters.fe_degree, DataOut<dim>::curved_inner_cells);

    const std::string filename_pressure =
      "sol_deg" + Utilities::int_to_string(parameters.fe_degree,1)
      + "_" + wave_equation_op->Name()
      + "_case" + Utilities::int_to_string(parameters.initial_cases,1)
      + "_ref" +Utilities::int_to_string(parameters.n_refinements,1)
      + "_step" + Utilities::int_to_string (time_control.get_output_step_number(), 3);

    {
      std::ostringstream filename;
      filename << "output/"
               << filename_pressure;
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        filename << "_Proc"
                 << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      filename << ".vtu";

      std::ofstream output_pressure (filename.str().c_str());
      data_out.write_vtu (output_pressure);
    }


    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1 &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          {
            std::ostringstream filename;
            filename << filename_pressure
                     << "_Proc"
                     << i
                     << ".vtu";

            filenames.push_back(filename.str().c_str());
          }
        std::string master_name = "output/" + filename_pressure + ".pvtu";
        std::ofstream master_output (master_name.c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }


    Vector<double> norm_per_cell_p (triangulation.n_active_cells());

    ComponentSelectFunction<dim> pressure_select(dim, dim+1);
    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       solutions,
                                       ZeroFunction<dim>(dim+1),
                                       norm_per_cell_p,
                                       QGauss<dim>(fe.degree+1),
                                       VectorTools::L2_norm,
                                       &pressure_select);
    double solution_mag = std::sqrt(Utilities::MPI::sum (norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));
    double solution_norm_p = 0.0, solution_norm_v = 0.0, solution_norm_p_post = 0.0;

    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       solutions,
                                       ExactSolution<dim>(dim+1,dim,time_control.get_time(),parameters.initial_cases,parameters.membrane_modes),
                                       norm_per_cell_p,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::L2_norm,
                                       &pressure_select);

    last_error_val = solution_norm_p = std::sqrt(Utilities::MPI::sum (norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    ComponentSelectFunction<dim> velocity_select(std::pair<unsigned int,unsigned int>(0U, dim), dim+1);
    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       solutions,
                                       ExactSolution<dim>(dim+1,-1,time_control.get_time(),parameters.initial_cases,parameters.membrane_modes),
                                       norm_per_cell_p,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::L2_norm,
                                       &velocity_select);
    solution_norm_v = std::sqrt(Utilities::MPI::sum (norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    // compute post pressure
    wave_equation_op->compute_post_pressure(solutions, tmp_solutions, post_pressure);
    VectorTools::integrate_difference (mapping,
                                       dof_handler_post_disp,
                                       post_pressure,
                                       ExactSolution<dim>(1, dim,time_control.get_time(),parameters.initial_cases,parameters.membrane_modes),
                                       norm_per_cell_p,
                                       QGauss<dim>(fe.degree+3),
                                       VectorTools::L2_norm);

    solution_norm_p_post = std::sqrt(Utilities::MPI::sum (norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    tmp_solutions = 0;
    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       tmp_solutions,
                                       ExactSolution<dim>(dim+1,dim,time_control.get_time(),parameters.initial_cases,parameters.membrane_modes),
                                       norm_per_cell_p,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::L2_norm,
                                       &pressure_select);
    solution_mag = std::sqrt(Utilities::MPI::sum (norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));

    if (parameters.cfl_stability_analysis)
      {
        if (first_error_val<0.0)
          {
            first_error_val = last_error_val;
            first_mangnitude_val = solution_mag;
          }
        if (last_error_val>100.0*first_error_val || last_error_val>1.5*first_mangnitude_val)
          time_control.set_time(parameters.final_time);
      }

    pcout << "   Time:"
          << std::fixed << std::setw(8) << std::setprecision(2) << time_control.get_time()
          << " , error p: "
          << std::scientific << std::setprecision(4) << std::setw(10) << solution_norm_p
          << " , error p post: "
          << std::scientific << std::setprecision(4) << std::setw(10) << solution_norm_p_post
          << " , error v: "
          << std::scientific << std::setprecision(4) << std::setw(10) << solution_norm_v
          << " , solution mag p: "
          << std::scientific << std::setprecision(4) << std::setw(10) << solution_mag
          << std::endl;

    pcout << "write output for time step " << time_control.get_step_number()
          << " at time " << std::fixed << std::setprecision(2) <<time_control.get_time()
          << std::endl;

  }



  template <int dim>
  double compute_time_step_size (const Triangulation<dim> &triangulation, Parameters &parameters)
  {
    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();
    double min_cell_diameter = std::numeric_limits<double>::max();
    double diameter = 0.0;

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          diameter = cell->minimum_vertex_distance() / input_materials()[cell->material_id()].speed;
          if (diameter < min_cell_diameter)
            min_cell_diameter = diameter;
        }

    return parameters.cfl_number * Utilities::MPI::min(min_cell_diameter, MPI_COMM_WORLD);
  }



  template<int dim>
  void WaveEquationProblem<dim>::run()
  {
    make_grid();

    // setup time control
    time_control.setup(parameters.final_time,
                       parameters.output_every_time,
                       compute_time_step_size(triangulation,parameters),
                       parameters.max_time_steps);

    pcout << "Time step size: " << time_control.get_time_step() << std::endl << std::endl;

    // determine wave equation operation, i.e. how to evaluate the integrals
    switch (parameters.integ_type)
      {
      case IntegratorType::expleuler:
      case IntegratorType::classrk4:
      case IntegratorType::lsrk45reg2:
      case IntegratorType::lsrk33reg2:
      case IntegratorType::lsrk45reg3:
      case IntegratorType::lsrk59reg2:
      case IntegratorType::ssprk:
      {
        if (parameters.fe_degree==1)
          wave_equation_op.reset(new WaveEquationOperation<dim,1>(time_control,parameters));
        else if (parameters.fe_degree==2)
          wave_equation_op.reset(new WaveEquationOperation<dim,2>(time_control,parameters));
        else if (parameters.fe_degree==3)
          wave_equation_op.reset(new WaveEquationOperation<dim,3>(time_control,parameters));
        else if (parameters.fe_degree==4)
          wave_equation_op.reset(new WaveEquationOperation<dim,4>(time_control,parameters));
        else if (parameters.fe_degree==5)
          wave_equation_op.reset(new WaveEquationOperation<dim,5>(time_control,parameters));
        else if (parameters.fe_degree==6)
          wave_equation_op.reset(new WaveEquationOperation<dim,6>(time_control,parameters));
        else if (parameters.fe_degree==7)
          wave_equation_op.reset(new WaveEquationOperation<dim,7>(time_control,parameters));
        else if (parameters.fe_degree==8)
          wave_equation_op.reset(new WaveEquationOperation<dim,8>(time_control,parameters));
        else if (parameters.fe_degree==9)
          wave_equation_op.reset(new WaveEquationOperation<dim,9>(time_control,parameters));
        else if (parameters.fe_degree==10)
          wave_equation_op.reset(new WaveEquationOperation<dim,10>(time_control,parameters));
        else if (parameters.fe_degree==11)
          wave_equation_op.reset(new WaveEquationOperation<dim,11>(time_control,parameters));
        else if (parameters.fe_degree==12)
          wave_equation_op.reset(new WaveEquationOperation<dim,12>(time_control,parameters));
        else
          Assert (false, ExcNotImplemented());
        break;
      }
      case IntegratorType::ader:
      {
        if (parameters.fe_degree==1)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,1>(time_control,parameters));
        else if (parameters.fe_degree==2)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,2>(time_control,parameters));
        else if (parameters.fe_degree==3)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,3>(time_control,parameters));
        else if (parameters.fe_degree==4)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,4>(time_control,parameters));
        else if (parameters.fe_degree==5)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,5>(time_control,parameters));
        else if (parameters.fe_degree==6)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,6>(time_control,parameters));
        else if (parameters.fe_degree==7)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,7>(time_control,parameters));
        else if (parameters.fe_degree==8)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,8>(time_control,parameters));
        else if (parameters.fe_degree==9)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,9>(time_control,parameters));
        else if (parameters.fe_degree==10)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,10>(time_control,parameters));
        else if (parameters.fe_degree==11)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,11>(time_control,parameters));
        else if (parameters.fe_degree==12)
          wave_equation_op.reset(new WaveEquationOperationADER<dim,12>(time_control,parameters));
        else
          Assert (false, ExcNotImplemented());
        break;
      }
      case IntegratorType::ader_lts:
      {
        // after this call, the variable time_step is set to the biggest time_step of the LTS scheme
        if (parameters.fe_degree==1)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,1>(time_control,parameters));
        else if (parameters.fe_degree==2)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,2>(time_control,parameters));
        else if (parameters.fe_degree==3)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,3>(time_control,parameters));
        else if (parameters.fe_degree==4)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,4>(time_control,parameters));
        else if (parameters.fe_degree==5)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,5>(time_control,parameters));
        else if (parameters.fe_degree==6)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,6>(time_control,parameters));
        else if (parameters.fe_degree==7)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,7>(time_control,parameters));
        else if (parameters.fe_degree==8)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,8>(time_control,parameters));
        else if (parameters.fe_degree==9)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,9>(time_control,parameters));
        else if (parameters.fe_degree==10)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,10>(time_control,parameters));
        else if (parameters.fe_degree==11)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,11>(time_control,parameters));
        else if (parameters.fe_degree==12)
          wave_equation_op.reset(new WaveEquationOperationADERLTS<dim,12>(time_control,parameters));
        else
          Assert (false, ExcNotImplemented());
        break;
      }
      case IntegratorType::ader_adconfull:
      {
        if (parameters.fe_degree==1)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,1>(time_control,parameters));
        else if (parameters.fe_degree==2)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,2>(time_control,parameters));
        else if (parameters.fe_degree==3)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,3>(time_control,parameters));
        else if (parameters.fe_degree==4)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,4>(time_control,parameters));
        else if (parameters.fe_degree==5)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,5>(time_control,parameters));
        else if (parameters.fe_degree==6)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,6>(time_control,parameters));
        else if (parameters.fe_degree==7)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,7>(time_control,parameters));
        else if (parameters.fe_degree==8)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,8>(time_control,parameters));
        else if (parameters.fe_degree==9)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,9>(time_control,parameters));
        else if (parameters.fe_degree==10)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,10>(time_control,parameters));
        else if (parameters.fe_degree==11)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,11>(time_control,parameters));
        else if (parameters.fe_degree==12)
          wave_equation_op.reset(new WaveEquationOperationADERADCONFULL<dim,12>(time_control,parameters));
        else
          Assert (false, ExcNotImplemented());
        break;
      }
      default:
        Assert (false, ExcNotImplemented());
      }

    make_dofs();
    pcout << "   Time step size: " << time_control.get_time_step() << std::endl;


    // set initial conditions
    wave_equation_op->project_initial_field(solutions, ExactSolution<dim> (dim+1, -1, time_control.get_time(),parameters.initial_cases,parameters.membrane_modes));


    unsigned int n_refinements_left = parameters.n_adaptive_refinements;
    while (n_refinements_left > 0)
      {
        adapt_mesh();
        wave_equation_op->project_initial_field(solutions, ExactSolution<dim> (dim+1, -1, time_control.get_time(),parameters.initial_cases,parameters.membrane_modes));
        --n_refinements_left;
        if (n_refinements_left == 0)
          {
            Vector<double> error_per_cell(triangulation.n_active_cells());
            wave_equation_op->estimate_error(solutions, tmp_solutions, error_per_cell);
            maximal_cellwise_error_init =
              Utilities::MPI::max(error_per_cell.linfty_norm(), MPI_COMM_WORLD);
          }
      }

    // output initial fields
    output_results();

    // determine integrator, i.e. how to combine the state vectors
    std::shared_ptr<ExplicitIntegrator<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> > > integrator;
    switch (parameters.integ_type)
      {
      case IntegratorType::expleuler:
      {
        integrator.reset(new ExplicitEuler<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::classrk4:
      {
        integrator.reset(new ClassRK4<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::lsrk45reg2:
      {
        integrator.reset(new LowStorageRK45Reg2<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::lsrk33reg2:
      {
        integrator.reset(new LowStorageRK33Reg2<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::lsrk45reg3:
      {
        integrator.reset(new LowStorageRK45Reg3<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::lsrk59reg2:
      {
        integrator.reset(new LowStorageRK59Reg2<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::ssprk:
      {
        integrator.reset(new SSPRK<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >(4,8));
        break;
      }
      case IntegratorType::ader:
      case IntegratorType::ader_adconfull:
      {
        integrator.reset(new ArbitraryHighOrderDG<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      case IntegratorType::ader_lts:
      {
        integrator.reset(new ArbitraryHighOrderDGLTS<LinearAlgebra::distributed::Vector<value_type>,WaveEquationOperationBase<dim> >());
        break;
      }
      default:
        Assert (false, ExcNotImplemented());
      }

    Timer timer;
    double wtime = 0.0;
    double output_time = 0.0;
    while (!time_control.done())
      {
        time_control.advance_time_step();

        timer.restart();
        tmp_solutions.swap(solutions);

        integrator->perform_time_step(tmp_solutions,solutions,time_control.get_time_step(),*wave_equation_op);
        wtime += timer.wall_time();

        if (parameters.n_adaptive_refinements > 0)
          if (time_control.get_step_number() % parameters.adaptive_refinement_interval == 0)
            adapt_mesh();

        timer.restart();
        time_step_analysis(mapping, dof_handler, solutions, time_control.get_time());

        if (time_control.at_tick())
          output_results();
        output_time += timer.wall_time();
      }

    pcout << std::endl
          << "   Performed " << time_control.get_step_number() << " time steps."
          << std::endl;

    pcout << "   Average wallclock time per time step: "
          << wtime /  time_control.get_step_number() << "s, time per element: "
          << wtime/ time_control.get_step_number()/triangulation.n_active_cells()
          << "s" << std::endl;

    pcout << "   Spent " << output_time << " s on output";
    pcout << "   and   " << Utilities::MPI::max(wtime,MPI_COMM_WORLD) << " s on computations." << std::endl;

  }

  void run_cfl_stability_analysis(Parameters &parameters_in)
  {
    ConditionalOStream pcout(std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    double cfl_test = parameters_in.cfl_number;
    double cfl_closest_stable = -0.1;
    double cfl_closest_instable = 100.0;

    for (unsigned int i=0; i<12; ++i)
      {
        pcout<<"*******************************************************"<<std::endl;
        pcout<<"cfl "<<cfl_test *std::pow(parameters_in.fe_degree,1.5)<<" in iteration "<<i<<std::endl;
        pcout<<"*******************************************************"<<std::endl;
        pcout<<std::endl;


        // run the problem in 2D or 3D and record error parameters
        bool simu_was_stable = false;
        if (parameters_in.dimension == 2)
          {
            WaveEquationProblem<2> we_problem(parameters_in);
            we_problem.run ();
            simu_was_stable = we_problem.cfl_stable();
          }
        else if (parameters_in.dimension == 3)
          {
            WaveEquationProblem<3> we_problem(parameters_in);
            we_problem.run ();
            simu_was_stable = we_problem.cfl_stable();
          }

        if (simu_was_stable)
          cfl_closest_stable = cfl_test;
        else
          cfl_closest_instable = cfl_test;

        if (cfl_closest_stable<0.0)
          {
            if (cfl_test/std::pow(parameters_in.fe_degree,1.5)>0.15)
              cfl_test -= 0.1;
            else
              cfl_test /= 3.0;
          }
        else if (cfl_closest_instable>99.0)
          cfl_test += 0.05;
        else
          cfl_test = (cfl_closest_instable+cfl_closest_stable)/2.0;

        parameters_in.cfl_number = cfl_test;
      }

    pcout<<"*******************************************************"<<std::endl;
    pcout<<"Final results for the CFL stability analysis:"<<std::endl;
    pcout<<"The Courant number                "<<cfl_closest_instable *std::pow(parameters_in.fe_degree,1.5)<<" is instable"<<std::endl;
    pcout<<"The Courant number                "<<cfl_closest_stable *std::pow(parameters_in.fe_degree,1.5)<<" is stable"<<std::endl;
    pcout<<"The limit might be in the middle: "<<(cfl_closest_instable+cfl_closest_instable)*std::pow(parameters_in.fe_degree,1.5)/2.0<<std::endl;
    pcout<<"*******************************************************"<<std::endl;
    pcout<<std::endl;

    return;
  }
}

int main (int argc, char **argv)
{
  using namespace HDG_WE;
  using namespace dealii;

#ifdef __x86_64

  // on x86-64:
  // change mode for rounding: denormals are flushed to zero to avoid computing
  // on denormals which can slow down computations a lot.
#define MXCSR_DAZ (1 << 6)      /* Enable denormals are zero mode */
#define MXCSR_FTZ (1 << 15)     /* Enable flush to zero mode */

  unsigned int mxcsr = __builtin_ia32_stmxcsr ();
  mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
  __builtin_ia32_ldmxcsr (mxcsr);

#endif

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << std::endl
                    << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                    << DEAL_II_GIT_BRANCH << std::endl;
          std::cout << "Number of MPI ranks:         "
                    << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                    << std::endl << std::endl;
        }

      deallog.depth_console (0);

      std::string paramfile;
      if (argc>1)
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
              we_problem.run ();
            }
          else
            run_cfl_stability_analysis(parameters);
        }
      else if (parameters.dimension == 3)
        {
          if (!parameters.cfl_stability_analysis)
            {
              WaveEquationProblem<3> we_problem(parameters);
              we_problem.run ();
            }
          else
            run_cfl_stability_analysis(parameters);
        }
      else
        AssertThrow(false,
                    ExcMessage("Invalid dimension " + std::to_string(parameters.dimension)));


      // output of the used parameters to be able to rerun the simulation
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        parameters.output_parameters(std::cout);

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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
