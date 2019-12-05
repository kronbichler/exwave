// --------------------------------------------------------------------------
//
// Copyright (C) 2018 by the ExWave authors
//
// This file is part of the ExWave library.
//
// The ExWave library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version. The full text of the
// license can be found in the file LICENSE at the top level of the ExWave
// distribution.
//
// --------------------------------------------------------------------------

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/timer.h>

#include "../include/linearized_euler_operations.h"

namespace DG_Euler
{
  template <int fe_degree>
  struct QuadratureDegreeSelector
  {
    static constexpr unsigned int n = fe_degree + 2;
    //static constexpr unsigned int n = 3 * fe_degree / 2 + 1;
  };


  template <int dim, int fe_degree, typename Number>
  InverseMassMatrixData<dim,fe_degree,Number>::InverseMassMatrixData(const MatrixFree<dim,Number> &data)
    :
    phi(1, FEEvaluation<dim,fe_degree,fe_degree+1,dim+2,Number>(data, 0,
                                                                /*quadrature formula number*/1)),
    coefficients(phi[0].n_q_points),
    inverse(phi[0])
  {}

  template <int dim, int fe_degree, typename Number>
  InverseMassMatrixData<dim,fe_degree,Number>::InverseMassMatrixData(const InverseMassMatrixData &other)
    :
    phi(other.phi),
    coefficients(other.coefficients),
    inverse(phi[0])
  {}


  template<int dim, int fe_degree>
  LinearizedEulerOperation<dim,fe_degree>::~LinearizedEulerOperation()
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    Utilities::MPI::MinMaxAvg data;
    if (computing_times[2] > 0)
      {
        pcout << "Computing " << (std::size_t)computing_times[2]
              << " times:" << std::endl;
        pcout << "   Evaluate     : ";
        data = Utilities::MPI::min_max_avg(computing_times[0], MPI_COMM_WORLD);
        pcout << std::scientific << std::setw(9) << data.min
              << " (p" << std::setw(4) << data.min_index << ") "
              << std::setw(9) << data.avg
              <<  std::setw(9) << data.max
              << " (p" << std::setw(4) << data.max_index << ")" << std::endl;
        pcout << "   Inverse mass : ";
        data = Utilities::MPI::min_max_avg(computing_times[1], MPI_COMM_WORLD);
        pcout << std::scientific << std::setw(9) << data.min
              << " (p" << std::setw(4) << data.min_index << ") "
              << std::setw(9) << data.avg
              <<  std::setw(9) << data.max
              << " (p" << std::setw(4) << data.max_index << ")" << std::endl;
      }
    else
      {
        pcout<<" call of invmass in apply          "<< std::scientific << std::setw(4) << Utilities::MPI::max(computing_times[1], MPI_COMM_WORLD)<<std::endl;
        pcout<<" call of domain and faces in apply "<< std::scientific << std::setw(4) << Utilities::MPI::max(computing_times[0], MPI_COMM_WORLD)<<std::endl;
      }
  }


  template<int dim, int fe_degree>
  std::string LinearizedEulerOperation<dim,fe_degree>::Name()
  {
    return "Runge-Kutta";
  }

  template<int dim, int fe_degree>
  const MatrixFree<dim,typename LinearizedEulerOperationBase<dim>::value_type>
  &LinearizedEulerOperation<dim,fe_degree>::get_matrix_free() const
  {
    return data;
  }

  template<int dim, int fe_degree>
  HDG_WE::TimeControl &LinearizedEulerOperation<dim,fe_degree>::get_time_control() const
  {
    return time_control;
  }



  template<int dim, int fe_degree>
  LinearizedEulerOperation<dim,fe_degree>::
  LinearizedEulerOperation(HDG_WE::TimeControl &time_control_in,
                           Parameters &parameters_in)
    :
    time_control(time_control_in),
    parameters(parameters_in),
    computing_times(3)
  {}



  template<int dim, int fe_degree>
  void LinearizedEulerOperation<dim,fe_degree>::
  setup(const Mapping<dim>                         &mapping,
        const std::vector<const DoFHandler<dim> *> &dof_handlers)
  {
    AffineConstraints<value_type> dummy;
    dummy.close();
    std::vector<const AffineConstraints<value_type> *> constraints(dof_handlers.size(),&dummy);

    // Add a second quadrature formula that is used for computing the
    // integrals in post-processing, including the cross terms to the standard
    // DoFHandler.
    std::vector<Quadrature<1> > quadratures(2);
    quadratures[0] = QGauss<1>(QuadratureDegreeSelector<fe_degree>::n);
    quadratures[1] = QGauss<1>(fe_degree+1);

    typename MatrixFree<dim,value_type>::AdditionalData additional_data;
    //additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,value_type>::AdditionalData::none;
    additional_data.hold_all_faces_to_owned_cells = true;
    additional_data.overlap_communication_computation = false;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points |
                                            update_values);
    additional_data.mapping_update_flags_inner_faces = (update_JxW_values |
                                                        update_quadrature_points | update_normal_vectors |
                                                        update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_JxW_values |
                                                           update_quadrature_points | update_normal_vectors |
                                                           update_values);

    data.reinit(mapping,dof_handlers,constraints,quadratures,additional_data);

    mass_matrix_data.reset(new InverseMassMatrixData<dim,fe_degree,value_type>(data));
  }



  constexpr double GAMMA = 1.4;

  template <int dim, typename Number>
  Tensor<1,dim,Number>
  euler_velocity(const Tensor<1,dim+2,Number> &conserved_variables)
  {
    const Number inverse_density = 1./conserved_variables[0];
    Tensor<1,dim,Number> velocity;
    for (unsigned int d=0; d<dim; ++d)
      velocity[d] = conserved_variables[1+d] * inverse_density;
    return velocity;
  }

  template <int dim, typename Number>
  Number
  euler_pressure(const Tensor<1,dim+2,Number> &conserved_variables)
  {
    const Tensor<1,dim,Number> velocity = euler_velocity<dim>(conserved_variables);
    Number rhou_u = conserved_variables[1] * velocity[0];
    for (unsigned int d=1; d<dim; ++d)
      rhou_u += conserved_variables[1+d] * velocity[d];

    return (GAMMA - 1.) * (conserved_variables[dim+1] - 0.5 * rhou_u);
  }

  template <int dim, typename Number>
  Tensor<1,dim+2,Tensor<1,dim,Number>>
  euler_flux(const Tensor<1,dim+2,Number> &conserved_variables)
  {
    const Tensor<1,dim,Number> velocity = euler_velocity<dim>(conserved_variables);
    const Number pressure = euler_pressure<dim>(conserved_variables);

    Tensor<1,dim+2,Tensor<1,dim,Number>> flux;
    for (unsigned int d=0; d<dim; ++d)
      {
        flux[0][d] = conserved_variables[1+d];
        for (unsigned int e=0; e<dim; ++e)
          flux[e+1][d] = conserved_variables[e+1] * velocity[d];
        flux[d+1][d] += pressure;
        flux[dim+1][d] = velocity[d] * (conserved_variables[dim+1]+pressure);
      }
    return flux;
  }

  template <int n_components, int dim, typename Number>
  Tensor<1, n_components, Number> operator *
  (const Tensor<1,n_components,Tensor<1,dim,Number>> &matrix,
   const Tensor<1,dim,Number> &vector)
  {
    Tensor<1, n_components, Number> result;
    for (unsigned int d=0; d<n_components; ++d)
      result[d] = matrix[d] * vector;
    return result;
  }



  template<int dim, int fe_degree>
  void LinearizedEulerOperation<dim, fe_degree>::
  local_apply_domain(const MatrixFree<dim,value_type>                     &data,
                     LinearAlgebra::distributed::Vector<value_type>       &dst,
                     const LinearAlgebra::distributed::Vector<value_type> &src,
                     const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    FEEvaluation<dim,fe_degree,QuadratureDegreeSelector<fe_degree>::n,dim+2,value_type>
        fe_eval(data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(src, true, false);

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            fe_eval.submit_gradient(euler_flux<dim>(fe_eval.get_value(q)), q);
            //fe_eval.submit_value(..., q);
          }

        fe_eval.integrate_scatter(false, true, dst);
        //fe_eval.integrate_scatter(true, true, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LinearizedEulerOperation<dim,fe_degree>::
  local_apply_face (const MatrixFree<dim,value_type> &,
                    LinearAlgebra::distributed::Vector<value_type>       &dst,
                    const LinearAlgebra::distributed::Vector<value_type> &src,
                    const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,QuadratureDegreeSelector<fe_degree>::n,dim+2,value_type>
        fe_eval_m(this->data, true),
        fe_eval_p(this->data, false);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_p.reinit(face);
        fe_eval_p.gather_evaluate(src,true,false);

        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src,true,false);

        for (unsigned int q=0; q<fe_eval_p.n_q_points; ++q)
          {
            const auto u_m = fe_eval_m.get_value(q);
            const auto u_p = fe_eval_p.get_value(q);

            const auto velocity_m = euler_velocity<dim>(u_m);
            const auto velocity_p = euler_velocity<dim>(u_p);

            const auto pressure_m = euler_pressure<dim>(u_m);
            const auto pressure_p = euler_pressure<dim>(u_p);

            const auto flux_m = euler_flux<dim>(u_m);
            const auto flux_p = euler_flux<dim>(u_p);

            const auto normal = fe_eval_m.get_normal_vector(q);

            const auto lambda = 0.5 *
              std::sqrt(std::max(velocity_p.norm_square() + std::abs(GAMMA * pressure_p * (1./u_p[0])),
                                 velocity_m.norm_square() + std::abs(GAMMA * pressure_m * (1./u_m[0]))));

            const auto numerical_flux = (flux_m * normal + flux_p * normal) * 0.5 +
                lambda * (u_m - u_p);

            fe_eval_m.submit_value(numerical_flux, q);
            fe_eval_p.submit_value(-numerical_flux, q);
          }

        fe_eval_p.integrate_scatter(true,false,dst);
        fe_eval_m.integrate_scatter(true,false,dst);
      }
  }


  template <int dim, int fe_degree>
  void LinearizedEulerOperation<dim,fe_degree>::
  local_apply_boundary_face (const MatrixFree<dim,value_type> &,
                             LinearAlgebra::distributed::Vector<value_type>       &dst,
                             const LinearAlgebra::distributed::Vector<value_type> &src,
                             const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,QuadratureDegreeSelector<fe_degree>::n,dim+2,value_type>
        fe_eval_m(this->data, true);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src,true,false);

        for (unsigned int q=0; q<fe_eval_m.n_q_points; ++q)
          {
            const auto u_m = fe_eval_m.get_value(q);

            //const auto velocity_m = euler_velocity<dim>(u_m);

            //const auto pressure_m = euler_pressure<dim>(u_m);

            const auto flux_m = euler_flux<dim>(u_m);

            const auto normal = fe_eval_m.get_normal_vector(q);

            //VectorizedArray<value_type> lambda = 0.5 *
            //  std::sqrt(velocity_m.norm_square() + std::abs(GAMMA * pressure_m * (1./u_m[0])));

            const auto numerical_flux = flux_m * normal * 0.5;

            fe_eval_m.submit_value(numerical_flux, q);
          }
        fe_eval_m.integrate_scatter(true,false,dst);
      }
  }




  template<int dim, int fe_degree>
  void LinearizedEulerOperation<dim,fe_degree>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type> &,
                          LinearAlgebra::distributed::Vector<value_type>       &dst,
                          const LinearAlgebra::distributed::Vector<value_type> &src,
                          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        mass_matrix_data->phi[0].reinit(cell);
        mass_matrix_data->phi[0].read_dof_values(src);

        mass_matrix_data->inverse.fill_inverse_JxW_values(mass_matrix_data->coefficients);
        mass_matrix_data->inverse.apply(mass_matrix_data->coefficients, dim+2,
                                        mass_matrix_data->phi[0].begin_dof_values(),
                                        mass_matrix_data->phi[0].begin_dof_values());

        mass_matrix_data->phi[0].set_dof_values(dst);
      }
  }



  template<int dim, int fe_degree>
  void LinearizedEulerOperation<dim, fe_degree>::
  apply(const LinearAlgebra::distributed::Vector<value_type>  &src,
        LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    Timer timer;
    data.loop (&LinearizedEulerOperation<dim, fe_degree>::local_apply_domain,
               &LinearizedEulerOperation<dim, fe_degree>::local_apply_face,
               &LinearizedEulerOperation<dim, fe_degree>::local_apply_boundary_face,
               this, dst, src, true,
               MatrixFree<dim,value_type>::DataAccessOnFaces::values,
               MatrixFree<dim,value_type>::DataAccessOnFaces::values);
    computing_times[0] += timer.wall_time();

    timer.restart();
    data.cell_loop(&LinearizedEulerOperation<dim, fe_degree>::local_apply_mass_matrix,
                   this, dst, dst);
    computing_times[1] += timer.wall_time();

    computing_times[2] += 1.;
  }



  template <int dim, int fe_degree>
  void LinearizedEulerOperation<dim, fe_degree>::
  estimate_error(const LinearAlgebra::distributed::Vector<value_type> &solution,
                 LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                 Vector<double>                                       &error_estimate) const
  {
    (void)solution;
    (void)tmp_vector;
    (void)error_estimate;
    AssertThrow(false, ExcNotImplemented());
  }



  template<int dim, int fe_degree>
  void LinearizedEulerOperation<dim, fe_degree>::
  project_initial_field(LinearAlgebra::distributed::Vector<value_type> &solution,
                        const Function<dim>                            &function) const
  {
    const unsigned int n_q_points = FEEvaluation<dim,fe_degree,fe_degree+1,dim+2,value_type>::static_n_q_points;
    FEEvaluation<dim,fe_degree,fe_degree+1,dim+2,value_type> &phi = mass_matrix_data->phi[0];

    for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            Point<dim,VectorizedArray<value_type> > q_points = phi.quadrature_point(q);
            Tensor<1,dim+2,VectorizedArray<value_type> > rhs;
            for (unsigned int d=0; d<dim+2; ++d)
              for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
                {
                  Point<dim> q_point;
                  for (unsigned int e=0; e<dim; ++e)
                    q_point[e] = q_points[e][v];
                  rhs[d][v] = function.value(q_point,d);
                }
            phi.submit_value(rhs,q);
          }
        phi.integrate(true,false);

        mass_matrix_data->inverse.fill_inverse_JxW_values(mass_matrix_data->coefficients);
        mass_matrix_data->inverse.apply(mass_matrix_data->coefficients, dim+1,
                                        phi.begin_dof_values(),
                                        phi.begin_dof_values());
        phi.set_dof_values(solution);
      }
  }



  // explicit instaniation for all operators for space dimensions 2,3 and polynomial degrees 1,...,12
  template class LinearizedEulerOperation<2,1>;
  template class LinearizedEulerOperation<3,1>;

  template class LinearizedEulerOperation<2,2>;
  template class LinearizedEulerOperation<3,2>;

  template class LinearizedEulerOperation<2,3>;
  template class LinearizedEulerOperation<3,3>;

  template class LinearizedEulerOperation<2,4>;
  template class LinearizedEulerOperation<3,4>;

  template class LinearizedEulerOperation<2,5>;
  template class LinearizedEulerOperation<3,5>;

//  template class LinearizedEulerOperation<2,6>;
//  template class LinearizedEulerOperation<3,6>;

//  template class LinearizedEulerOperation<2,7>;
//  template class LinearizedEulerOperation<3,7>;

//  template class LinearizedEulerOperation<2,8>;
//  template class LinearizedEulerOperation<3,8>;

//  template class LinearizedEulerOperation<2,9>;
//  template class LinearizedEulerOperation<3,9>;

//  template class LinearizedEulerOperation<2,10>;
//  template class LinearizedEulerOperation<3,10>;

//  template class LinearizedEulerOperation<2,11>;
//  template class LinearizedEulerOperation<3,11>;

//  template class LinearizedEulerOperation<2,12>;
//  template class LinearizedEulerOperation<3,12>;

}
