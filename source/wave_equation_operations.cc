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

#include "../include/wave_equation_operations.h"

namespace HDG_WE
{

  template <int dim, int fe_degree, typename Number>
  InverseMassMatrixData<dim,fe_degree,Number>::InverseMassMatrixData(const MatrixFree<dim,Number> &data)
    :
    phi(1, FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,Number>(data)),
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
  WaveEquationOperation<dim,fe_degree>::~WaveEquationOperation()
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
        if (computing_times[4] > 0)
          {
            pcout << "   ADER CK step : ";
            data = Utilities::MPI::min_max_avg(computing_times[4], MPI_COMM_WORLD);
            pcout << std::scientific << std::setw(9) << data.min
                  << " (p" << std::setw(4) << data.min_index << ") "
                  << std::setw(9) << data.avg
                  <<  std::setw(9) << data.max
                  << " (p" << std::setw(4) << data.max_index << ")" << std::endl;
          }
        if (computing_times[5] > 0)
          {
            pcout << "   ADER 2nd step: ";
            data = Utilities::MPI::min_max_avg(computing_times[5], MPI_COMM_WORLD);
            pcout << std::scientific << std::setw(9) << data.min
                  << " (p" << std::setw(4) << data.min_index << ") "
                  << std::setw(9) << data.avg
                  <<  std::setw(9) << data.max
                  << " (p" << std::setw(4) << data.max_index << ")" << std::endl;
          }
        if (computing_times[6] > 0)
          {
            pcout << "   ADER inv mass: ";
            data = Utilities::MPI::min_max_avg(computing_times[6], MPI_COMM_WORLD);
            pcout << std::scientific << std::setw(9) << data.min
                  << " (p" << std::setw(4) << data.min_index << ") "
                  << std::setw(9) << data.avg
                  <<  std::setw(9) << data.max
                  << " (p" << std::setw(4) << data.max_index << ")" << std::endl;
          }
      }
    else
      {
        pcout<<" call of invmass in apply          "<< std::scientific << std::setw(4) << Utilities::MPI::max(computing_times[1], MPI_COMM_WORLD)<<std::endl;
        pcout<<" call of domain and faces in apply "<< std::scientific << std::setw(4) << Utilities::MPI::max(computing_times[0], MPI_COMM_WORLD)<<std::endl;
      }
  }


  template<int dim, int fe_degree>
  std::string WaveEquationOperation<dim,fe_degree>::Name()
  {
    return "Runge-Kutta";
  }

  template<int dim, int fe_degree>
  const MatrixFree<dim,typename WaveEquationOperationBase<dim>::value_type> &WaveEquationOperation<dim,fe_degree>::get_matrix_free() const
  {
    return data;
  }

  template<int dim, int fe_degree>
  TimeControl &WaveEquationOperation<dim,fe_degree>::get_time_control() const
  {
    return time_control;
  }

  template<int dim, int fe_degree>
  void WaveEquationOperation<dim,fe_degree>::apply_ader (const LinearAlgebra::distributed::Vector<value_type> &,
                                                         LinearAlgebra::distributed::Vector<value_type> &) const
  {
    AssertThrow(false,ExcNotImplemented());
  }


  template<int dim, int fe_degree>
  unsigned int WaveEquationOperation<dim,fe_degree>::cluster_id(unsigned int) const
  {
    return -1;
  }

  template<int dim, int fe_degree>
  typename WaveEquationOperationBase<dim>::value_type WaveEquationOperation<dim,fe_degree>::time_step(unsigned int ) const
  {
    return time_control.get_time_step();
  }

  template<int dim, int fe_degree>
  typename WaveEquationOperationBase<dim>::value_type WaveEquationOperation<dim,fe_degree>::speed_of_sound(int cell_index, int vect_index) const
  {
    return speeds[cell_index][vect_index];
  }


  template<int dim, int fe_degree>
  std::string WaveEquationOperationADER<dim,fe_degree>::Name()
  {
    return "ADER";
  }


  template<int dim, int fe_degree>
  std::string WaveEquationOperationADERADCONFULL<dim,fe_degree>::Name()
  {
    return "ADERADCONFULL";
  }


  template<int dim, int fe_degree>
  std::string WaveEquationOperationADERLTS<dim,fe_degree>::Name()
  {
    return "ADERLTS";
  }

  template<int dim, int fe_degree>
  unsigned int WaveEquationOperationADERLTS<dim,fe_degree>::cluster_id(unsigned int cell) const
  {
    return cluster_manager.cell_cluster_ids[cell];
  }

  template<int dim, int fe_degree>
  typename WaveEquationOperation<dim,fe_degree>::value_type WaveEquationOperationADERLTS<dim,fe_degree>::time_step(unsigned int cell) const
  {
    return cluster_manager.get_cell_time_step(cell);
  }

  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::communicate_flux_memory() const
  {
    flux_memory.compress(VectorOperation::add);
    return;
  }


  template<int dim, int fe_degree>
  WaveEquationOperation<dim,fe_degree>::
  WaveEquationOperation(TimeControl &time_control_in, Parameters &parameters_in)
    :
    time_control(time_control_in),
    parameters(parameters_in),
    computing_times(23)
  {}



  template<int dim, int fe_degree>
  void WaveEquationOperation<dim,fe_degree>::
  setup(const MappingQGeneric<dim>                 &mapping,
        const std::vector<const DoFHandler<dim> *> &dof_handlers,
        const std::vector<Material>                &mats,
        const std::vector<unsigned int>            &vectorization_categories)
  {
    AffineConstraints<value_type> dummy;
    dummy.close();
    std::vector<const AffineConstraints<value_type> *> constraints(dof_handlers.size(),&dummy);

    // Add a second quadrature formula that is used for computing the
    // integrals in post-processing, including the cross terms to the standard
    // DoFHandler.
    std::vector<Quadrature<1> > quadratures(2);
    quadratures[0] = QGauss<1>(fe_degree+1);
    quadratures[1] = QGauss<1>(fe_degree+2);
    const unsigned int n_steps = fe_degree / 2;
    for (unsigned int q=0; q<n_steps; ++q)
      quadratures.push_back(QGauss<1>(fe_degree-q*2-1));

    typename MatrixFree<dim,value_type>::AdditionalData additional_data;
    //additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,value_type>::AdditionalData::partition_partition;
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
    additional_data.initialize_mapping = false;
    additional_data.cell_vectorization_category = vectorization_categories;
    additional_data.cell_vectorization_categories_strict = true;

    data.reinit(mapping,dof_handlers,constraints,quadratures,additional_data);

    std::vector<types::global_dof_index> renumbering;
    data.renumber_dofs(renumbering, 0);
    const_cast<DoFHandler<dim> *>(dof_handlers[0])->renumber_dofs(renumbering);
    additional_data.initialize_mapping = true;
    data.reinit(mapping,dof_handlers,constraints,quadratures,additional_data);

    mass_matrix_data.reset(new InverseMassMatrixData<dim,fe_degree,value_type>(data));
    reset_data_vectors(mats);
  }



  template<int dim, int fe_degree>
  WaveEquationOperationADER<dim,fe_degree>::WaveEquationOperationADER(TimeControl &time_control_in, Parameters &parameters_in)
    : WaveEquationOperation<dim,fe_degree>(time_control_in,parameters_in)
  {}


  template<int dim, int fe_degree>
  WaveEquationOperationADER<dim,fe_degree>::~WaveEquationOperationADER()
  {}



  template<int dim, int fe_degree>
  void WaveEquationOperationADER<dim,fe_degree>::
  setup(const MappingQGeneric<dim>                 &mapping,
        const std::vector<const DoFHandler<dim> *> &dof_handlers,
        const std::vector<Material>                &mats,
        const std::vector<unsigned int>            &vectorization_categories)
  {
    // call base class setup
    WaveEquationOperation<dim,fe_degree>::setup(mapping,dof_handlers,mats,vectorization_categories);

    const unsigned int n_steps = 1+fe_degree/2;
    shape_infos.resize(n_steps);
    shape_infos_embed.resize(n_steps);
    for (unsigned int k=0; k<n_steps; ++k)
      {
        QGauss<1> quad(fe_degree+1-k*2);
        FE_DGQArbitraryNodes<1> fe(quad);
        shape_infos[k].reinit(quad, fe);
        if (k>0)
          {
            QGauss<1> quad_embed(fe_degree+1-(k-1)*2);
            shape_infos_embed[k-1].reinit(quad_embed, fe);
            // we want to use the shape Hessians in shape_infos_embed to store
            // a particular projection operation: When performing a projection
            // from a higher polynomial degree to a lower one, we want to
            // perform the following operations:
            //
            // loop over quadrature points, multiply by q-weight on high degree
            // integrate loop going from high degree to low degree
            // loop over new points, multiply by inverse q-weight on low degree
            //
            // This particular setup is a projection step on the unit element
            // for the special case of Lagrange polynomials where most of the
            // interpolation matrices are unit matrices when applying the
            // inverse mass matrix.
            //
            // We want to do this operation in one single step, which can be
            // done by putting the weights into the interpolation matrix into
            // shape_hessians_eo. This is for matrices in even-odd format, but
            // luckily the quadrature weights are also symmetric about the
            // center of the cell
            const unsigned int stride = (quad_embed.size()+1)/2;
            AssertDimension(stride*quad.size(), shape_infos_embed[k-1].shape_values_eo.size());
            AssertDimension(stride*quad.size(), shape_infos_embed[k-1].shape_hessians_eo.size());
            for (unsigned int i=0; i<quad.size(); ++i)
              for (unsigned int q=0; q<stride; ++q)
                shape_infos_embed[k-1].shape_hessians_eo[i*stride+q] =
                  shape_infos_embed[k-1].shape_values_eo[i*stride+q] * (quad_embed.weight(q) / quad.weight(i));
          }
      }

    // initialize vector for temporary values
    this->data.initialize_dof_vector(tempsrc);
  }



  template<int dim, int fe_degree>
  WaveEquationOperationADERADCONFULL<dim,fe_degree>::
  WaveEquationOperationADERADCONFULL(TimeControl &time_control_in,
                                     Parameters  &parameters_in)
    : WaveEquationOperation<dim,fe_degree>(time_control_in,parameters_in)
  {}



  template<int dim, int fe_degree>
  void WaveEquationOperationADERADCONFULL<dim,fe_degree>::
  setup(const MappingQGeneric<dim>                 &mapping,
        const std::vector<const DoFHandler<dim> *> &dof_handlers,
        const std::vector<Material>                &mats,
        const std::vector<unsigned int>            &vectorization_categories)
  {
    // call base class setup
    WaveEquationOperation<dim,fe_degree>::setup(mapping,dof_handlers,mats,vectorization_categories);

    this->data.initialize_dof_vector(tempsrc);
  }



  template<int dim, int fe_degree>
  WaveEquationOperationADERLTS<dim,fe_degree>::
  WaveEquationOperationADERLTS(TimeControl &time_control_in, Parameters &parameters_in)
    : WaveEquationOperationADER<dim,fe_degree>(time_control_in,parameters_in)
  {}



  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  setup(const MappingQGeneric<dim>                 &mapping,
        const std::vector<const DoFHandler<dim> *> &dof_handlers,
        const std::vector<Material>                &mats,
        const std::vector<unsigned int> &)
  {
    // call base class setup
    std::vector<unsigned int> new_vec_categors;
    cluster_manager.propose_cluster_categorization(new_vec_categors,
                                                   dof_handlers[0]->get_triangulation(),
                                                   dof_handlers,
                                                   this->parameters.max_n_clusters,
                                                   this->parameters.max_diff_clusters,
                                                   this->parameters.cfl_number);
    WaveEquationOperationADER<dim,fe_degree>::setup(mapping,dof_handlers,mats,new_vec_categors);

    // initialize flux memory
    this->data.initialize_dof_vector(flux_memory);
    cluster_manager.state = flux_memory;
    if (this->parameters.use_ader_post)
      cluster_manager.improvedgraddiv = flux_memory;

    cluster_manager.setup(*this);
  }



  template <int dim, int fe_degree>
  void
  WaveEquationOperation<dim,fe_degree>::reset_data_vectors(const std::vector<Material> mats)
  {
    densities.resize(data.n_macro_cells()+data.n_ghost_cell_batches());
    speeds.resize(data.n_macro_cells()+data.n_ghost_cell_batches());

    for (unsigned int i=0; i<data.n_macro_cells()+data.n_ghost_cell_batches(); ++i)
      {
        densities[i] = 1.;
        speeds[i] = 1.;
        for (unsigned int v=0; v<data.n_components_filled(i); ++v)
          {
            densities[i][v] = mats[data.get_cell_iterator(i,v)->material_id()].density;
            speeds[i][v] = mats[data.get_cell_iterator(i,v)->material_id()].speed;
          }
      }
  }



  template<int dim, int fe_degree>
  void WaveEquationOperation<dim, fe_degree>::
  local_apply_domain(const MatrixFree<dim,value_type>                     &data,
                     LinearAlgebra::distributed::Vector<value_type>       &dst,
                     const LinearAlgebra::distributed::Vector<value_type> &src,
                     const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data, 0, 0, 0);
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> pressure(data, 0, 0, dim);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        evaluate_cell(velocity,pressure,src,cell);
        velocity.distribute_local_to_global (dst);
        pressure.distribute_local_to_global (dst);
      }
  }



  template <int dim, int fe_degree>
  void
  WaveEquationOperation<dim,fe_degree>::
  evaluate_cell(FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> &phi_v,
                FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type>   &phi_p,
                const LinearAlgebra::distributed::Vector<value_type>   &src,
                const unsigned int                                      cell) const
  {
    // It is faster to evaluate values of the vector-valued velocity and
    // gradients of the scalar pressure than divergence of velocity and
    // values of pressure
    phi_v.reinit(cell);
    phi_v.gather_evaluate (src, true, false);

    phi_p.reinit(cell);
    phi_p.gather_evaluate(src, false, true);

    const VectorizedArray<value_type> rho = this->densities[cell];
    const VectorizedArray<value_type> rho_inv = 1./this->densities[cell];
    const VectorizedArray<value_type> c_sq = this->speeds[cell]*this->speeds[cell];

    for (unsigned int q=0; q<phi_v.n_q_points; ++q)
      {
        const Tensor<1,dim,VectorizedArray<value_type> >
        pressure_gradient = phi_p.get_gradient(q);

        phi_p.submit_gradient(rho*c_sq*phi_v.get_value(q), q);
        phi_v.submit_value(-rho_inv*pressure_gradient,q);
      }

    phi_v.integrate (true, false);
    phi_p.integrate (false, true);
  }



  template <int dim, int fe_degree>
  void
  WaveEquationOperation<dim,fe_degree>::
  local_apply_face (const MatrixFree<dim,value_type> &,
                    LinearAlgebra::distributed::Vector<value_type>       &dst,
                    const LinearAlgebra::distributed::Vector<value_type> &src,
                    const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    // There is some overhead in the methods in FEEvaluation, so it is faster
    // to combine pressure and velocity in the same object and just combine
    // them at the level of quadrature points
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi_neighbor(this->data, false, 0, 0, 0);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      evaluate_inner_face(phi,phi_neighbor,src,face,1.0,&dst);
  }



  template <int dim, int fe_degree>
  void WaveEquationOperation<dim,fe_degree>::
  evaluate_inner_face(FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi,
                      FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_neighbor,
                      const LinearAlgebra::distributed::Vector<value_type>         &src,
                      const unsigned int                                            face,
                      const value_type                                              boundary_fac,
                      LinearAlgebra::distributed::Vector<value_type>               *dst) const
  {
    phi.reinit(face);
    phi.gather_evaluate(src, true, false);
    const VectorizedArray<value_type> rho_plus = phi.read_cell_data(densities);
    const VectorizedArray<value_type> rho_inv_plus = 1./rho_plus;
    const VectorizedArray<value_type> c_plus = phi.read_cell_data(speeds);
    const VectorizedArray<value_type> c_sq_plus = c_plus * c_plus;
    const VectorizedArray<value_type> tau_plus = 1./c_plus/rho_plus;

    phi_neighbor.reinit(face);
    phi_neighbor.gather_evaluate(src, true, false);
    const VectorizedArray<value_type> rho_minus = phi_neighbor.read_cell_data(densities);
    const VectorizedArray<value_type> rho_inv_minus = 1./rho_minus;
    const VectorizedArray<value_type> c_minus = phi_neighbor.read_cell_data(speeds);
    const VectorizedArray<value_type> c_sq_minus = c_minus * c_minus;
    const VectorizedArray<value_type> tau_minus = 1./c_minus/rho_minus;

    const VectorizedArray<value_type> tau_inv = 1./(tau_plus + tau_minus);

    AssertDimension(phi.n_q_points, data.get_n_q_points_face(0));

    for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
        Tensor<1,dim+1,VectorizedArray<value_type> > val_plus = phi.get_value(q);
        Tensor<1,dim+1,VectorizedArray<value_type> > val_minus = phi_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = phi.get_normal_vector(q);
        VectorizedArray<value_type> normal_v_plus = val_plus[0]*normal[0];
        VectorizedArray<value_type> normal_v_minus = -val_minus[0]*normal[0];
        for (unsigned int d=1; d<dim; ++d)
          {
            normal_v_plus += val_plus[d] * normal[d];
            normal_v_minus -= val_minus[d] * normal[d];
          }

        VectorizedArray<value_type> lambda = tau_inv*( normal_v_plus
                                                       + normal_v_minus
                                                       + tau_plus * val_plus[dim]
                                                       + tau_minus * val_minus[dim]);
        VectorizedArray<value_type> pres_diff_plus = (val_plus[dim]-lambda)*rho_inv_plus;
        VectorizedArray<value_type> pres_diff_minus = (val_minus[dim]-lambda)*rho_inv_minus;
        for (unsigned int d=0; d<dim; ++d)
          {
            val_plus[d] = boundary_fac*pres_diff_plus*normal[d];
            val_minus[d] = -boundary_fac*pres_diff_minus*normal[d];
          }
        val_plus[dim] = boundary_fac * c_sq_plus * rho_plus * (-normal_v_plus + tau_plus * (lambda - val_plus[dim]));
        val_minus[dim] = boundary_fac * c_sq_minus * rho_minus * (-normal_v_minus + tau_minus * (lambda - val_minus[dim]));

        phi.submit_value(val_plus, q);
        phi_neighbor.submit_value(val_minus, q);
      }
    // dst is not the null pointer -> directly write into the result
    if (dst != nullptr)
      {
        phi.integrate_scatter(true,false,*dst);
        phi_neighbor.integrate_scatter(true,false,*dst);
      }
    // else do the full interpolation, a special function
    else
      {
        phi.integrate(true,false);
        phi_neighbor.integrate(true,false);
      }
  }



  template <int dim, int fe_degree>
  void WaveEquationOperation<dim,fe_degree>::
  local_apply_boundary_face (const MatrixFree<dim,value_type> &,
                             LinearAlgebra::distributed::Vector<value_type>       &dst,
                             const LinearAlgebra::distributed::Vector<value_type> &src,
                             const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);
    for (unsigned int face=face_range.first; face<face_range.second; face++)
      evaluate_boundary_face(phi,src,face,1.0,&dst);
  }



  template<int dim, int fe_degree>
  void WaveEquationOperation<dim,fe_degree>::
  evaluate_boundary_face(FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type>  &phi,
                         const LinearAlgebra::distributed::Vector<value_type>          &src,
                         const unsigned int                                            face,
                         const value_type                                              boundary_fac,
                         LinearAlgebra::distributed::Vector<value_type>               *dst) const
  {
    phi.reinit(face);
    phi.gather_evaluate(src,true,false);

    const VectorizedArray<value_type> rho = phi.read_cell_data(densities);
    const VectorizedArray<value_type> rho_inv = 1./rho;
    const VectorizedArray<value_type> c = phi.read_cell_data(speeds);
    const VectorizedArray<value_type> c_sq = phi.read_cell_data(speeds)*phi.read_cell_data(speeds);
    const VectorizedArray<value_type> tau = 1./phi.read_cell_data(speeds)/phi.read_cell_data(this->densities);

    const int boundary_id = int(this->data.get_boundary_id(face));

    for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > normal = phi.get_normal_vector(q);
        Tensor<1,dim+1,VectorizedArray<value_type> > val_plus = phi.get_value(q);
        VectorizedArray<value_type> p_plus = val_plus[dim];
        VectorizedArray<value_type> normal_v_plus = val_plus[0] * normal[0];
        for (unsigned int d=1; d<dim; ++d)
          normal_v_plus += val_plus[d] * normal[d];

        VectorizedArray<value_type> lambda;
        switch (boundary_id)
          {
          case 1: // soft wall - normal velocity component is zero
          {
            lambda = 1./tau*normal_v_plus+p_plus;
            break;
          }
          case 2: // hard wall - pressure is zero
          {
            lambda = VectorizedArray<value_type>();
            break;
          }
          case 3: // absorbing wall - mimics an open domain by the first order absorbing condition
          {
            lambda = tau/(tau+1./c/rho)*p_plus + 1./(tau+1./c/rho)*normal_v_plus;
            break;
          }
          default:
            Assert(false,ExcMessage("set your boundary ids correctly: 1 - soft wall, 2 - hard wall, 3 - first order ABC"));
          }

        for (unsigned int d=0; d<dim; ++d)
          val_plus[d] = boundary_fac*(p_plus-lambda)*rho_inv*normal[d];
        val_plus[dim] = boundary_fac*c_sq*rho*(-normal_v_plus+tau*(lambda - p_plus));

        phi.submit_value(val_plus,q);
      }
    if (dst != nullptr)
      phi.integrate_scatter(true,false,*dst);
    else
      phi.integrate(true,false);
  }



  template<int dim, int fe_degree>
  void WaveEquationOperation<dim,fe_degree>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type> &,
                          LinearAlgebra::distributed::Vector<value_type>       &dst,
                          const LinearAlgebra::distributed::Vector<value_type> &src,
                          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
#ifdef GAUSS_POINTS_VECTOR_OPERATION
    constexpr unsigned int dofs_per_component = Utilities::pow(fe_degree+1,dim);
#endif
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        mass_matrix_data->phi[0].reinit(cell);
        mass_matrix_data->phi[0].read_dof_values(src);

        mass_matrix_data->inverse.fill_inverse_JxW_values(mass_matrix_data->coefficients);
#ifdef GAUSS_POINTS_VECTOR_OPERATION
        for (unsigned int i=0; i<dofs_per_component; ++i)
          for (unsigned int d=0; d<dim+1; ++d)
            mass_matrix_data->phi[0].begin_dof_values()[d*dofs_per_component+i] *= mass_matrix_data->coefficients[i];
#else
        mass_matrix_data->inverse.apply(mass_matrix_data->coefficients, dim+1,
                                        mass_matrix_data->phi[0].begin_dof_values(),
                                        mass_matrix_data->phi[0].begin_dof_values());
#endif

        mass_matrix_data->phi[0].set_dof_values(dst);
      }
  }



  template<int dim, int fe_degree>
  void WaveEquationOperation<dim, fe_degree>::
  apply(const LinearAlgebra::distributed::Vector<value_type>  &src,
        LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    Timer timer;
    data.loop (&WaveEquationOperation<dim, fe_degree>::local_apply_domain,
               &WaveEquationOperation<dim, fe_degree>::local_apply_face,
               &WaveEquationOperation<dim, fe_degree>::local_apply_boundary_face,
               this, dst, src, true,
               MatrixFree<dim,value_type>::DataAccessOnFaces::values,
               MatrixFree<dim,value_type>::DataAccessOnFaces::values);
    computing_times[0] += timer.wall_time();

    timer.restart();
    data.cell_loop(&WaveEquationOperation<dim, fe_degree>::local_apply_mass_matrix,
                   this, dst, dst);
    computing_times[1] += timer.wall_time();

    computing_times[2] += 1.;
  }



  template<int dim, int fe_degree>
  void WaveEquationOperationADERADCONFULL<dim, fe_degree>::
  apply_ader(const LinearAlgebra::distributed::Vector<value_type>  &src,
             LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    LinearAlgebra::distributed::Vector<value_type> tempvals;
    tempvals.reinit(dst);
    this->tempsrc=0;

    double dt = this->time_control.get_time_step();

    // k=0 contribution
    //{
    this->data.loop (&WaveEquationOperation<dim, fe_degree>::local_apply_domain,
                     &WaveEquationOperation<dim, fe_degree>::local_apply_face,
                     &WaveEquationOperation<dim, fe_degree>::local_apply_boundary_face,
                     static_cast<const WaveEquationOperation<dim,fe_degree>*>(this), dst, src,
                     true, MatrixFree<dim,value_type>::DataAccessOnFaces::values,
                     MatrixFree<dim,value_type>::DataAccessOnFaces::values);
    this->data.cell_loop(&WaveEquationOperation<dim, fe_degree>::local_apply_mass_matrix,
                         static_cast<const WaveEquationOperation<dim,fe_degree>*>(this), dst, dst);
    this->tempsrc.sadd(1.0,-dt,dst);
    tempvals.equ(-1.,dst);
    //}

    // all remaining contributions
    double fac = -0.5*dt*dt;
    for (int k=1; k<=fe_degree; ++k)
      {
        this->data.loop (&WaveEquationOperation<dim, fe_degree>::local_apply_domain,
                         &WaveEquationOperation<dim, fe_degree>::local_apply_face,
                         &WaveEquationOperation<dim, fe_degree>::local_apply_boundary_face,
                         static_cast<const WaveEquationOperation<dim,fe_degree>*>(this), dst, tempvals,
                         true, MatrixFree<dim,value_type>::DataAccessOnFaces::values,
                         MatrixFree<dim,value_type>::DataAccessOnFaces::values);
        this->data.cell_loop(&WaveEquationOperation<dim, fe_degree>::local_apply_mass_matrix,
                             static_cast<const WaveEquationOperation<dim,fe_degree>*>(this), dst, dst);

        this->tempsrc.sadd(1.0,-fac,dst);
        tempvals.equ(-1.,dst);
        dst = 0.;

        fac *= -dt/(k+2);
      }

    dst = this->tempsrc;
  }


  namespace PostProcessingHelper
  {
    template <int dim, int fe_degree, typename Number>
    class MatrixPostProcessing
    {
    public:
      MatrixPostProcessing (FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> &phi)
        :
        phi(phi)
      {};

      void vmult(VectorizedArray<Number> *dst,
                 const VectorizedArray<Number> *src) const
      {
        // This implements the constant-coefficient Laplacian
        // (nabla phi, nabla phi)
        // Since that is a singular system on an element (= pure Neumann
        // problem), we need to constrain one degree of freedom. We pick the
        // first one. This corresponds to setting the row and column in that
        // entry to zero and putting one on the diagonal, which is implemented
        // by (temporarily) modifying the input vector and finally the result
        // vector. The value of the constant needs to be adjusted after solving
        // the linear system.
        VectorizedArray<Number> tmp_value = src[0];
        const_cast<VectorizedArray<Number> *>(src)[0] = 0.;
        phi.evaluate(src, false, true);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(false, true, dst);
        dst[0] = tmp_value;
        const_cast<VectorizedArray<Number> *>(src)[0] = tmp_value;
      }

      void precondition(VectorizedArray<Number> *dst,
                        const VectorizedArray<Number> *src) const
      {
        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
          dst[i] = src[i];
      }

    private:
      FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> &phi;
    };
  }


  template <int dim, int fe_degree>
  void WaveEquationOperation<dim, fe_degree>::
  compute_post_pressure(const LinearAlgebra::distributed::Vector<value_type> &solution,
                        LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                        LinearAlgebra::distributed::Vector<value_type>            &post_pressure_vector) const
  {
    WaveEquationOperation<dim,fe_degree>::apply(solution, tmp_vector);

    FEEvaluation<dim,fe_degree,fe_degree+2,dim,value_type> velocity(data, 0, 1, 0);
    FEEvaluation<dim,fe_degree,fe_degree+2,1,value_type> pressure(data, 0, 1, dim);
    FEEvaluation<dim,fe_degree+1,fe_degree+2,1,value_type> post_pressure(data, 1, 1, 0);
    AlignedVector<VectorizedArray<value_type> > rhs(post_pressure.dofs_per_cell),
                  result(post_pressure.dofs_per_cell);
    PostProcessingHelper::MatrixPostProcessing<dim,fe_degree+1,value_type> matrix(post_pressure);
    IterativeHelper::SolverCGvect<VectorizedArray<value_type> > cg_solver(post_pressure.dofs_per_cell,
        std::max(1e-20, static_cast<double>(1000.*std::sqrt(std::numeric_limits<value_type>::min()))), std::max(1e-11,1e2*std::numeric_limits<value_type>::epsilon()), 6*post_pressure.dofs_per_cell);

    for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
      {
        post_pressure.reinit(cell);

        velocity.reinit(cell);
        velocity.read_dof_values(tmp_vector);
        velocity.evaluate(true, false);

        // compute rhs (nabla phi, -rho v)
        const VectorizedArray<value_type> rho = densities[cell];
        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          post_pressure.submit_gradient(-rho * velocity.get_value(q), q);

        post_pressure.integrate(false, true, rhs.begin());

        // since the linear system is singular, we need to remove one
        // equation. We remove the first one by setting it to zero in
        // accordance with the changes to the matrix.
        rhs[0] = 0;
        cg_solver.solve(matrix, result.begin(), rhs.begin());

        // Adjust the constant offset. We compare the average as we got it
        // from solving the linear system with the correct average in the
        // 'solution[dim]' vector.
        pressure.reinit(cell);
        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          pressure.submit_value(make_vectorized_array(value_type(1.0)), q);
        const VectorizedArray<value_type> volume = pressure.integrate_value();

        pressure.read_dof_values(solution);
        pressure.evaluate(true, false);
        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          pressure.submit_value(pressure.get_value(q), q);
        const VectorizedArray<value_type> correct_pressure = pressure.integrate_value();

        post_pressure.evaluate(result.begin(), true, false);
        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          pressure.submit_value(post_pressure.get_value(q), q);
        const VectorizedArray<value_type> preliminary_pressure = pressure.integrate_value();
        const VectorizedArray<value_type> adjust_value = (correct_pressure-preliminary_pressure)/volume;

        // Apply the constant offset and write the final result
        for (unsigned int i=0; i<result.size(); ++i)
          post_pressure.begin_dof_values()[i] = result[i] + adjust_value;

        post_pressure.set_dof_values(post_pressure_vector);
      }
  }



  namespace
  {
    template <int dim, typename Number>
    Tensor<1,dim,Number> make_tensor(const Tensor<1,dim,Number> &in)
    {
      return in;
    }

    template <typename Number>
    Tensor<1,1,Number> make_tensor(const Number &in)
    {
      Tensor<1,1,Number> out;
      out[0] = in;
      return out;
    }
  }



  template <int dim, int fe_degree>
  void WaveEquationOperation<dim, fe_degree>::
  estimate_error(const LinearAlgebra::distributed::Vector<value_type> &solution,
                 LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                 Vector<double>                                       &post_pressure_vector) const
  {
    WaveEquationOperation<dim,fe_degree>::apply(solution, tmp_vector);

    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data, 0, 0, 0);
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> pressure(data, 0, 0, dim);

    for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
      {
        velocity.reinit(cell);
        velocity.gather_evaluate(tmp_vector, true, false);

        pressure.reinit(cell);
        pressure.gather_evaluate(solution, false, true);

        // Define the error as the square difference between the original
        // pressure gradient and the reconstructed pressure
        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          pressure.submit_value((pressure.get_gradient(q)-make_tensor(velocity.get_value(q))).norm_square(), q);
        const VectorizedArray<value_type> errors = pressure.integrate_value();
        for (unsigned int v=0; v<data.n_components_filled(cell); ++v)
          post_pressure_vector(data.get_cell_iterator(cell, v)->active_cell_index()) = std::sqrt(errors[v]);
      }
  }



  template<int dim, int fe_degree>
  void WaveEquationOperation<dim, fe_degree>::
  project_initial_field(LinearAlgebra::distributed::Vector<value_type> &solution,
                        const Function<dim>                            &function) const
  {
    const unsigned int n_q_points = FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type>::static_n_q_points;
    FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi = mass_matrix_data->phi[0];

    for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            Point<dim,VectorizedArray<value_type> > q_points = phi.quadrature_point(q);
            Tensor<1,dim+1,VectorizedArray<value_type> > rhs;
            for (unsigned int d=0; d<dim+1; ++d)
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



  template<int dim, int fe_degree>
  void WaveEquationOperationADER<dim,fe_degree>::
  local_apply_firstader_domain(const MatrixFree<dim,value_type> &,
                               LinearAlgebra::distributed::Vector<value_type>         &dst,
                               const LinearAlgebra::distributed::Vector<value_type>   &src,
                               const std::pair<unsigned int,unsigned int>                 &cell_range) const
  {
    // for calculation of higher spatial derivatives
    //{
    FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval = this->mass_matrix_data->phi[0];
    //}

    // cell loop
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        integrate_taylor_cauchykovalewski(cell,phi_eval,src,this->time_control.get_time_step(),0.0,0.0,dst);
        phi_eval.set_dof_values(dst);
      }
  }



  template <int dim, int fe_degree>
  void WaveEquationOperationADER<dim,fe_degree>::
  integrate_taylor_cauchykovalewski(const unsigned int                                        cell,
                                    FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval,
                                    const LinearAlgebra::distributed::Vector<value_type>     &src,
                                    const value_type                                          t2,
                                    const value_type                                          t1,
                                    const value_type                                          te,
                                    const LinearAlgebra::distributed::Vector<value_type>  &recongraddiv) const
  {
    const unsigned int n_q_points = FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type>::static_n_q_points;

    // and material coefficients
    const VectorizedArray<value_type> rho = this->densities[cell];
    const VectorizedArray<value_type> rho_inv = 1./this->densities[cell];
    const VectorizedArray<value_type> c_sq = this->speeds[cell]*this->speeds[cell];
    //}

    // create container for quass point pressure and velocity
    // contributions. vcontrib will be set to zero automatically through the
    // Tensor<1,dim> constructor, but pcontrib must be set manually -> this is
    // done in the first loop below
    VectorizedArray<value_type>                contrib[(dim+1)*n_q_points];

    // init cell
    phi_eval.reinit(cell);

    phi_eval.gather_evaluate (src, true, !(this->parameters.use_ader_post), false);
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        // contribution from k=0
        const Tensor<1,dim+1,VectorizedArray<value_type> > v_and_p = phi_eval.get_value(q);
        for (unsigned int d=0; d<dim; ++d)
          contrib[d*n_q_points+q] = (t2-t1)*v_and_p[d];
        contrib[dim*n_q_points+q] = (t2-t1)*v_and_p[dim];

        if (!(this->parameters.use_ader_post))
          {
            Tensor<1,dim+1,VectorizedArray<value_type> > v_and_p_next;
            const Tensor<1,dim+1,Tensor<1,dim,VectorizedArray<value_type> > > v_and_p_grad = phi_eval.get_gradient(q);
            v_and_p_next[dim] = VectorizedArray<value_type>();
            for (unsigned int d=0; d<dim; ++d)
              {
                v_and_p_next[d] = -rho_inv*v_and_p_grad[dim][d];
                v_and_p_next[dim] -= c_sq*rho*v_and_p_grad[d][d];
              }
            // add contribution from k=1
            for (unsigned int d=0; d<dim; ++d)
              contrib[d*n_q_points+q] += ((t2-te)*(t2-te)-(t1-te)*(t1-te))*0.5*v_and_p_next[d];
            contrib[dim*n_q_points+q] +=  ((t2-te)*(t2-te)-(t1-te)*(t1-te))*0.5*v_and_p_next[dim];

            // submit for further evaluation
            if (this->parameters.spectral_evaluation)
              {
                for (unsigned int d=0; d<dim+1; ++d)
                  phi_eval.begin_values()[q+d*n_q_points] = -v_and_p_next[d];
              }
            else
              phi_eval.submit_value(-v_and_p_next,q);
          }
      }

    if (this->parameters.use_ader_post)
      {
        phi_eval.gather_evaluate(recongraddiv, true, false, false);

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            Tensor<1,dim+1,VectorizedArray<value_type> > v_and_p_next = phi_eval.get_value(q);

            // add contribution from k=1
            for (unsigned int d=0; d<dim; ++d)
              contrib[d*n_q_points+q] +=  ((t2-te)*(t2-te)-(t1-te)*(t1-te))*0.5*v_and_p_next[d];
            contrib[dim*n_q_points+q] +=  ((t2-te)*(t2-te)-(t1-te)*(t1-te))*0.5*v_and_p_next[dim];

            // submit for further evaluation
            if (this->parameters.spectral_evaluation)
              {
                for (unsigned int d=0; d<dim+1; ++d)
                  phi_eval.begin_values()[q+d*n_q_points] = -v_and_p_next[d];
              }
            else
              phi_eval.submit_value(-v_and_p_next,q);
          }
      }

    this->mass_matrix_data->inverse.fill_inverse_JxW_values(this->mass_matrix_data->coefficients);

    // all following contributions can be looped
    integrate_taylor_cauchykovalewski_step<0,true>(cell, phi_eval, phi_eval.begin_values(),
                                                   t1-te, t2-te, contrib);

    // this operation corresponds to three steps:
    // phi_eval.submit_value();
    // phi_eval.integrate();
    // inverse.apply();
    this->mass_matrix_data->inverse.
    transform_from_q_points_to_basis(dim+1, contrib, phi_eval.begin_dof_values());
  }



  template <int dim, int fe_degree>
  template <int step_no, bool add_into_contrib>
  void WaveEquationOperationADER<dim,fe_degree>::
  integrate_taylor_cauchykovalewski_step(const unsigned int                                        cell,
                                         FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval,
                                         VectorizedArray<value_type>                              *spectral_array,
                                         const value_type                                          tstart,
                                         const value_type                                          tend,
                                         VectorizedArray<value_type>                              *contrib) const
  {
    // and material coefficients
    const VectorizedArray<value_type> rho = this->densities[cell];
    const VectorizedArray<value_type> rho_inv = 1./this->densities[cell];
    const VectorizedArray<value_type> c_sq = this->speeds[cell]*this->speeds[cell];
    //}

    value_type fac = -0.5;
    for (int k=2; k<=step_no+2; ++k)
      fac /= -(k+1);
    value_type fac_t = fac*(Utilities::fixed_power<step_no+3>(tend) - Utilities::fixed_power<step_no+3>(tstart));

    constexpr int reduce_step = step_no/2;
    constexpr int reduce_degree_by = 2 * reduce_step;
    bool at_reduce_step = ((step_no - 2 * reduce_step) == 1) && (this->parameters.spectral_evaluation==true);
    constexpr int my_degree = reduce_step>0 ? fe_degree-reduce_degree_by : fe_degree;
    constexpr unsigned int n_q_points = Utilities::pow(my_degree+1,dim);

    if (this->parameters.spectral_evaluation)
      {
        internal::EvaluatorTensorProduct<internal::evaluate_evenodd, dim, my_degree+1, my_degree+1, VectorizedArray<value_type> >
        eval(AlignedVector<VectorizedArray<value_type> >(),
             shape_infos[reduce_step].shape_gradients_collocation_eo,
             AlignedVector<VectorizedArray<value_type> >());
        if (this->data.get_mapping_info().get_cell_type(cell) == internal::MatrixFreeFunctions::cartesian)
          {
            eval.template gradients<0,true,false>(spectral_array, spectral_array);
            if (dim > 1)
              eval.template gradients<1,true,false>(spectral_array+n_q_points,
                                                    spectral_array+n_q_points);
            if (dim > 2)
              eval.template gradients<2,true,false>(spectral_array+2*n_q_points,
                                                    spectral_array+2*n_q_points);
            if (dim > 1)
              eval.template gradients<1,true,false>(spectral_array+dim*n_q_points,
                                                    spectral_array+(dim+1)*n_q_points);
            if (dim > 2)
              eval.template gradients<2,true,false>(spectral_array+dim*n_q_points,
                                                    spectral_array+(dim+2)*n_q_points);
            // derivative in 0 direction will overwrite the old content of
            // spectral array, so must do it after the other components of the
            // gradient...
            eval.template gradients<0,true,false>(spectral_array+dim*n_q_points,
                                                  spectral_array+dim*n_q_points);
            const Tensor<2,dim,VectorizedArray<value_type> > &jac =
              this->data.get_mapping_info().cell_data[0].jacobians
              [0][this->data.get_mapping_info().cell_data[0].data_index_offsets[cell]];
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                VectorizedArray<value_type> v_divergence = spectral_array[q] * jac[0][0];
                for (unsigned int d=1; d<dim; ++d)
                  v_divergence += spectral_array[q+d*n_q_points] * jac[d][d];

                if (!add_into_contrib)
                  {
                    for (unsigned int d=0; d<dim; ++d)
                      contrib[d*n_q_points+q] = (fac_t*rho_inv)*spectral_array[q+(dim+d)*n_q_points]*jac[d][d];
                    contrib[dim*n_q_points+q] = (fac_t*c_sq*rho)*v_divergence;
                  }
                else
                  {
                    for (unsigned int d=0; d<dim; ++d)
                      contrib[d*n_q_points+q] += (fac_t*rho_inv)*spectral_array[q+(dim+d)*n_q_points]*jac[d][d];
                    contrib[dim*n_q_points+q] += (fac_t*c_sq*rho)*v_divergence;
                  }

                for (unsigned int d=0; d<dim; ++d)
                  spectral_array[q+d*n_q_points] = rho_inv*spectral_array[q+(dim+d)*n_q_points]*jac[d][d];
                spectral_array[q+dim*n_q_points] = (c_sq*rho)*v_divergence;
              }
          }
        else
          {
            // go backwards to be able to write back into the same array
            for (int d=dim; d>=0; --d)
              {
                if (dim > 1)
                  eval.template gradients<1,true,false>(spectral_array+d*n_q_points,
                                                        spectral_array+(d*dim+1)*n_q_points);
                if (dim > 2)
                  eval.template gradients<2,true,false>(spectral_array+d*n_q_points,
                                                        spectral_array+(d*dim+2)*n_q_points);
                eval.template gradients<0,true,false>(spectral_array+d*n_q_points,
                                                      spectral_array+d*dim*n_q_points);
              }
            const unsigned int stride = (this->data.get_mapping_info().get_cell_type(cell) ==
                                         internal::MatrixFreeFunctions::general) ? 1 : 0;
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                constexpr int index = reduce_step>0 ? (reduce_step+1) : 0;
                const Tensor<2,dim,VectorizedArray<value_type> > &jac =
                  this->data.get_mapping_info().cell_data[index].jacobians
                  [0][this->data.get_mapping_info().cell_data[index].data_index_offsets[cell]+q*stride];
                AssertDimension(n_q_points,
                                this->data.get_mapping_info().cell_data[index].descriptor[0].quadrature.size());

                VectorizedArray<value_type> v_divergence = VectorizedArray<value_type>();
                for (unsigned int d=0; d<dim; ++d)
                  for (unsigned int e=0; e<dim; ++e)
                    v_divergence += spectral_array[q+(d*dim+e)*n_q_points] * jac[d][e];
                VectorizedArray<value_type> p_grad[dim];
                for (unsigned int d=0; d<dim; ++d)
                  {
                    p_grad[d] = spectral_array[q+dim*dim*n_q_points] * jac[d][0];
                    for (unsigned int e=1; e<dim; ++e)
                      p_grad[d] += spectral_array[q+(dim*dim+e)*n_q_points] * jac[d][e];
                  }

                if (!add_into_contrib)
                  {
                    for (unsigned int d=0; d<dim; ++d)
                      contrib[d*n_q_points+q] = (fac_t*rho_inv)*p_grad[d];
                    contrib[dim*n_q_points+q] = (fac_t*c_sq*rho)*v_divergence;
                  }
                else
                  {
                    for (unsigned int d=0; d<dim; ++d)
                      contrib[d*n_q_points+q] += (fac_t*rho_inv)*p_grad[d];
                    contrib[dim*n_q_points+q] += (fac_t*c_sq*rho)*v_divergence;
                  }

                for (unsigned int d=0; d<dim; ++d)
                  spectral_array[q+d*n_q_points] = rho_inv*p_grad[d];
                spectral_array[q+dim*n_q_points] = (c_sq*rho)*v_divergence;
              }
          }
      }
    else
      {
        // integrate over element
        phi_eval.integrate(true,false);

        // apply inverse mass matrix
        //{
        this->mass_matrix_data->inverse.apply(this->mass_matrix_data->coefficients, dim+1,
                                              phi_eval.begin_dof_values(),
                                              phi_eval.begin_dof_values());
        //}

        // evaulate this phi at the gauss points
        phi_eval.evaluate(false,true);

        // sum over all integration points
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            // get the gauss point values
            Tensor<1,dim+1,Tensor<1,dim,VectorizedArray<value_type> > > phi_gradient =
              phi_eval.get_gradient(q);

            // calculate contributions
            for (unsigned int d=0; d<dim; ++d)
              contrib[d*n_q_points+q] += (fac_t*rho_inv)*phi_gradient[dim][d];
            for (unsigned int d=0; d<dim; ++d)
              contrib[dim*n_q_points+q] += (fac_t*c_sq*rho)*phi_gradient[d][d];

            // evaluate things phi_k+1 needs
            Tensor<1,dim+1,VectorizedArray<value_type> > temp;
            for (unsigned int d=0; d<dim; ++d)
              {
                temp[d] = rho_inv*phi_gradient[dim][d];
                temp[dim] += c_sq*rho*phi_gradient[d][d];
              }
            phi_eval.submit_value(temp,q);
          }
      }

    if (step_no < fe_degree-2)
      {
        if (at_reduce_step)
          {
            // project contribution from higher degree to the lower degree
            constexpr int next_degree = my_degree >= 2 ? my_degree-2 : 1;
            internal::FEEvaluationImplBasisChange<internal::evaluate_evenodd,dim,next_degree+1,my_degree+1,dim+1,VectorizedArray<value_type>,VectorizedArray<value_type> >::do_backward(shape_infos_embed[reduce_step].shape_hessians_eo, false, spectral_array, spectral_array);
            VectorizedArray<value_type> next_contrib_array[Utilities::pow(next_degree+1,dim)*(dim+1)];

            // run Taylor-Cauchy-Kovalewski at lower degree
            this->template integrate_taylor_cauchykovalewski_step<(step_no<fe_degree-2 ? step_no+1 : step_no),false>
                                                                  (cell, phi_eval, spectral_array, tstart, tend, next_contrib_array);

            // interpolation correction to the higher degree contribution
            internal::FEEvaluationImplBasisChange<internal::evaluate_evenodd,dim,next_degree+1,my_degree+1,dim+1,VectorizedArray<value_type>,VectorizedArray<value_type> >::do_forward(shape_infos_embed[reduce_step].shape_values_eo, next_contrib_array, spectral_array);
            for (unsigned int q=0; q<n_q_points*(dim+1); ++q)
                   contrib[q] += spectral_array[q];
          }
        else
          {
            this->template integrate_taylor_cauchykovalewski_step<(step_no<fe_degree-2 ? step_no+1 : step_no),true>
                                                                  (cell, phi_eval, spectral_array, tstart, tend, contrib);
          }
      }
  }


  template <int dim, int fe_degree>
  void WaveEquationOperationADER<dim,fe_degree>::
  local_apply_ader_face (const MatrixFree<dim,value_type> &,
                         LinearAlgebra::distributed::Vector<value_type>       &dst,
                         const LinearAlgebra::distributed::Vector<value_type> &src,
                         const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    // basically the same as local_apply_face, but different signs in some places

    // There is some overhead in the methods in FEEvaluation, so it is faster
    // to combine pressure and velocity in the same object and just combine
    // them at the level of quadrature points
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi_neighbor(this->data, false, 0, 0, 0);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
           this->evaluate_inner_face(phi,phi_neighbor, src, face, -1.0, &dst);
  }



  template <int dim, int fe_degree>
  void WaveEquationOperationADER<dim,fe_degree>::
  local_apply_ader_boundary_face (const MatrixFree<dim,value_type> &,
                                  LinearAlgebra::distributed::Vector<value_type>       &dst,
                                  const LinearAlgebra::distributed::Vector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
           this->evaluate_boundary_face(phi, src, face, -1.0, &dst);
  }



  template<int dim, int fe_degree>
  void WaveEquationOperationADER<dim,fe_degree>::
  local_apply_secondader_domain(const MatrixFree<dim,value_type> &,
                                LinearAlgebra::distributed::Vector<value_type>       &dst,
                                const LinearAlgebra::distributed::Vector<value_type> &src,
                                const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(this->data, 0, 0, 0);
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> pressure(this->data, 0, 0, dim);

    // now: combine face and element stuff
    // cell loop
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        // get all cell quanitites
        //{
        // velocity
        velocity.reinit(cell);
        velocity.gather_evaluate (src, true, false);

        // pressure
        pressure.reinit(cell);
        pressure.gather_evaluate(src, false, true);

        // and material coefficients
        const VectorizedArray<value_type> rho = this->densities[cell];
        const VectorizedArray<value_type> rho_inv = 1./this->densities[cell];
        const VectorizedArray<value_type> c_sq = this->speeds[cell]*this->speeds[cell];
        //}

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            const Tensor<1,dim,VectorizedArray<value_type> > pressure_gradient = pressure.get_gradient(q);
            pressure.submit_gradient(-rho*c_sq*velocity.get_value(q), q);
            velocity.submit_value(rho_inv*pressure_gradient, q);
          }

        velocity.integrate_scatter (true, false, dst);

        pressure.integrate_scatter(false, true, dst);
      }
  }



  template<int dim, int fe_degree>
  void WaveEquationOperationADER<dim, fe_degree>::
  apply_ader(const LinearAlgebra::distributed::Vector<value_type>  &src,
             LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {

    Timer timer;
    if (this->parameters.use_ader_post)
      WaveEquationOperation<dim,fe_degree>::apply(src,tempsrc);
    this->computing_times[3] += timer.wall_time();

    // first ader step
    timer.restart();
    this->data.cell_loop (&WaveEquationOperationADER<dim, fe_degree>::local_apply_firstader_domain,
                          this, tempsrc, src);
    this->computing_times[4] += timer.wall_time();
    timer.restart();
    this->data.loop (&WaveEquationOperationADER<dim, fe_degree>::local_apply_secondader_domain,
                     &WaveEquationOperationADER<dim, fe_degree>::local_apply_ader_face,
                     &WaveEquationOperationADER<dim, fe_degree>::local_apply_ader_boundary_face,
                     this, dst, tempsrc, true,
                     MatrixFree<dim,value_type>::DataAccessOnFaces::values,
                     MatrixFree<dim,value_type>::DataAccessOnFaces::values);
    this->computing_times[5] += timer.wall_time();

    // inverse mass matrix
    timer.restart();
    this->data.cell_loop(&WaveEquationOperation<dim, fe_degree>::local_apply_mass_matrix,
                         static_cast<const WaveEquationOperation<dim,fe_degree>*>(this), dst, dst);
    this->computing_times[6] += timer.wall_time();

    // timinig
    //    this->computing_times[1] += timer.wall_time();
    //    this->computing_times[2] += 1.;

  }



  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_firstader_domain(const MatrixFree<dim,value_type> &,
                               LinearAlgebra::distributed::Vector<value_type>         &dst,
                               const LinearAlgebra::distributed::Vector<value_type>   &src,
                               const std::pair<unsigned int,unsigned int>                 &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval = this->mass_matrix_data->phi[0];

    // cell loop
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        if (cluster_manager.is_evaluate_cell(cell))
          {
            value_type t1 = cluster_manager.get_t1();
            value_type t2 = cluster_manager.get_t2();
            value_type te = cluster_manager.get_te(cell);

            this->integrate_taylor_cauchykovalewski(cell,phi_eval,src,t2,t1,te,cluster_manager.improvedgraddiv);
            phi_eval.set_dof_values(dst);
          }
      } // for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  }



  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_secondader_domain(const MatrixFree<dim,value_type> &,
                                LinearAlgebra::distributed::Vector<value_type>         &dst,
                                const LinearAlgebra::distributed::Vector<value_type>   &src,
                                const std::pair<unsigned int,unsigned int>             &cell_range) const
  {
    // for calculation of higher spatial derivatives
    //{
    const unsigned int n_q_points = FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type>::static_n_q_points;
    FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval = this->mass_matrix_data->phi[0];
    FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> help_eval(this->data); // for memory variable and update of src
    //}

    // cell loop
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        if (cluster_manager.is_update_cell(cell))
          {
            value_type dt = cluster_manager.get_dt();
            this->integrate_taylor_cauchykovalewski(cell,phi_eval,src,dt,0.,0.,cluster_manager.improvedgraddiv);

            // the standard business analog to local_apply_firstaderlts is done
            // now comes the update!
            {
              phi_eval.evaluate (true, true, false);

              const VectorizedArray<value_type> rho = this->densities[cell];
              const VectorizedArray<value_type> rho_inv = 1./this->densities[cell];
              const VectorizedArray<value_type> c_sq = this->speeds[cell]*this->speeds[cell];

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const Tensor<1,dim+1,Tensor<1,dim,VectorizedArray<value_type> > > v_and_p_grad = phi_eval.get_gradient(q);
                  const Tensor<1,dim+1,VectorizedArray<value_type> > v_and_p = phi_eval.get_value(q);

                  Tensor<1,dim+1,VectorizedArray<value_type> > temp_value;
                  for (unsigned int d=0; d<dim; ++d)
                         temp_value[d] = rho_inv*v_and_p_grad[dim][d];
                  phi_eval.submit_value(temp_value,q);

                  Tensor<1,dim+1,Tensor<1,dim,VectorizedArray<value_type> > > temp_gradient;
                  for (unsigned int d=0; d<dim; ++d)
                         temp_gradient[dim][d] = -rho*c_sq*v_and_p[d];

                  phi_eval.submit_gradient(temp_gradient,q);
                }

              phi_eval.integrate (true, true);

              // add memory variable
              unsigned int dofs_per_cell = phi_eval.dofs_per_cell;
              help_eval.reinit(cell);
              if (cluster_manager.is_fluxmemory_considered)
                {
                  help_eval.read_dof_values(flux_memory);
                  for (unsigned j=0; j<dofs_per_cell; ++j)
                         for (unsigned int d=0; d<dim+1; ++d)
                      {
                        phi_eval.begin_dof_values()[d*dofs_per_cell+j] += help_eval.begin_dof_values()[d*dofs_per_cell+j];
                        help_eval.begin_dof_values()[d*dofs_per_cell+j] = 0.;
                      }
                  help_eval.set_dof_values(flux_memory); // tell the flux_memory variable, that some of its values are reset
                }

              // add face contribution (stored in dst) and reset dst to zero
              help_eval.read_dof_values(dst, 0);
              for (unsigned j=0; j<dofs_per_cell; ++j)
                     for (unsigned int d=0; d<dim+1; ++d)
                  {
                    phi_eval.begin_dof_values()[d*dofs_per_cell+j] += help_eval.begin_dof_values()[d*dofs_per_cell+j];
                    help_eval.begin_dof_values()[d*dofs_per_cell+j] = 0.;
                  }
              help_eval.set_dof_values(dst);

              // apply inverse mass matrix
              this->mass_matrix_data->inverse.apply(this->mass_matrix_data->coefficients, dim+1,
                                                    phi_eval.begin_dof_values(),
                                                    phi_eval.begin_dof_values());
              //}
              phi_eval.set_dof_values(dst);
            }
          }
      } // for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  }



  // does nothing
  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_dummy_domain(const MatrixFree<dim,value_type> &,
                           LinearAlgebra::distributed::Vector<value_type> &,
                           const LinearAlgebra::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int> &) const
  {}



  template <int dim, int fe_degree>
  void
  WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_ader_face (const MatrixFree<dim,value_type> &,
                         LinearAlgebra::distributed::Vector<value_type>        &dst,
                         const LinearAlgebra::distributed::Vector<value_type>  &src,
                         const std::pair<unsigned int,unsigned int>            &face_range) const
  {
    // There is some overhead in the methods in FEEvaluation, so it is faster
    // to combine pressure and velocity in the same object and just combine
    // them at the level of quadrature points
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi_neighbor(this->data, false, 0, 0, 0);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        if (cluster_manager.is_evaluate_face(face))
          {
            this->evaluate_inner_face(phi, phi_neighbor, src, face, -1.0, nullptr);

            // get bitsets from cluster manager
            phi.distribute_local_to_global(dst, 0, cluster_manager.get_phi_to_dst(face));
            phi.distribute_local_to_global(flux_memory, 0, cluster_manager.get_phi_to_fluxmemory(face));
            phi_neighbor.distribute_local_to_global(dst, 0, cluster_manager.get_phi_neighbor_to_dst(face));
            phi_neighbor.distribute_local_to_global(flux_memory, 0, cluster_manager.get_phi_neighbor_to_fluxmemory(face));
          }
      }
  }

  template <int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_ader_boundary_face (const MatrixFree<dim,value_type> &,
                                  LinearAlgebra::distributed::Vector<value_type>       &dst,
                                  const LinearAlgebra::distributed::Vector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>               &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        if(cluster_manager.is_evaluate_face(face))
          {
            this->evaluate_boundary_face(phi, src, face, -1.0 ,nullptr);

            // write only to the cell who is active
            phi.distribute_local_to_global(dst, 0, cluster_manager.get_phi_to_dst(face));
          }
      }
  }

  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::evaluate_cells_and_faces_first_ader(
    const LinearAlgebra::distributed::Vector<value_type>  &src,
    LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    this->data.cell_loop (&WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_firstader_domain,
                          this, this->tempsrc, src, true);

    // evaluate faces for these cells
    this->data.loop (&WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_dummy_domain,
                     &WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_ader_face,
                     &WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_ader_boundary_face,
                     this, dst, this->tempsrc);
  }


  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::evaluate_cells_second_ader(
    const LinearAlgebra::distributed::Vector<value_type>  &src,
    LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    this->data.cell_loop (&WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_secondader_domain,
                          this, dst, src);
  }

  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  apply_ader(const LinearAlgebra::distributed::Vector<value_type>  &src,
             LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    cluster_manager.perform_time_step(*this,src,dst);
  }

  template <int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::reconstruct_div_grad(const LinearAlgebra::distributed::Vector<value_type>  &src,
      LinearAlgebra::distributed::Vector<value_type>        &dst) const
  {
    this->data.loop (&WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_postprocessing_domain,
                     &WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_postprocessing_face,
                     &WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_postprocessing_boundary_face,
                     this, dst, src);

    this->data.cell_loop(&WaveEquationOperationADERLTS<dim, fe_degree>::local_apply_postprocessing_mass_matrix,
                         this, dst, dst);
  }

  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim, fe_degree>::
  local_apply_postprocessing_domain(const MatrixFree<dim,value_type> &,
                                    LinearAlgebra::distributed::Vector<value_type>         &dst,
                                    const LinearAlgebra::distributed::Vector<value_type>   &src,
                                    const std::pair<unsigned int,unsigned int>                 &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(this->data, 0, 0, 0);
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> pressure(this->data, 0, 0, dim);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        if (cluster_manager.is_evaluate_cell(cell))
          {
            this->evaluate_cell(velocity,pressure,src,cell);
            velocity.set_dof_values(dst);
            pressure.set_dof_values (dst);
          }
      }
  }

  template <int dim, int fe_degree>
  void
  WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_postprocessing_face (const MatrixFree<dim,value_type> &,
                                   LinearAlgebra::distributed::Vector<value_type>        &dst,
                                   const LinearAlgebra::distributed::Vector<value_type>  &src,
                                   const std::pair<unsigned int,unsigned int>                &face_range) const
  {
    // There is some overhead in the methods in FEEvaluation, so it is faster
    // to combine pressure and velocity in the same object and just combine
    // them at the level of quadrature points
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi_neighbor(this->data, false, 0, 0, 0);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        if (cluster_manager.is_evaluate_face(face))
          {
            this->evaluate_inner_face(phi, phi_neighbor, src, face, 1.0, nullptr);
            phi.distribute_local_to_global(dst, 0, cluster_manager.get_phi_to_dst(face));
            phi_neighbor.distribute_local_to_global(dst, 0, cluster_manager.get_phi_neighbor_to_dst(face));
          }
      }
  }

  template <int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_postprocessing_boundary_face (const MatrixFree<dim,value_type> &,
                                            LinearAlgebra::distributed::Vector<value_type>       &dst,
                                            const LinearAlgebra::distributed::Vector<value_type> &src,
                                            const std::pair<unsigned int,unsigned int>               &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> phi(this->data, true, 0, 0, 0);
    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        if (cluster_manager.is_evaluate_face(face))
          {
            this->evaluate_boundary_face(phi, src, face, 1.0, nullptr);
            phi.distribute_local_to_global(dst, 0, cluster_manager.get_phi_to_dst(face));
          }
      }
  }

  template<int dim, int fe_degree>
  void WaveEquationOperationADERLTS<dim,fe_degree>::
  local_apply_postprocessing_mass_matrix(const MatrixFree<dim,value_type> &,
                                         LinearAlgebra::distributed::Vector<value_type>        &dst,
                                         const LinearAlgebra::distributed::Vector<value_type>  &src,
                                         const std::pair<unsigned int,unsigned int>    &cell_range) const
  {
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        if (cluster_manager.is_evaluate_cell(cell))
          {
            this->mass_matrix_data->phi[0].reinit(cell);
            this->mass_matrix_data->phi[0].read_dof_values(src);

            this->mass_matrix_data->inverse.fill_inverse_JxW_values(this->mass_matrix_data->coefficients);
            this->mass_matrix_data->inverse.apply(this->mass_matrix_data->coefficients, dim+1,
                                                  this->mass_matrix_data->phi[0].begin_dof_values(),
                                                  this->mass_matrix_data->phi[0].begin_dof_values());

            this->mass_matrix_data->phi[0].set_dof_values(dst);
          }
      }
  }


  // explicit instaniation for all operators for space dimensions 2,3 and polynomial degrees 1,...,12
  template class WaveEquationOperation<2,1>;
  template class WaveEquationOperation<3,1>;
  template class WaveEquationOperationADER<2,1>;
  template class WaveEquationOperationADER<3,1>;
  template class WaveEquationOperationADERLTS<2,1>;
  template class WaveEquationOperationADERLTS<3,1>;
  template class WaveEquationOperationADERADCONFULL<2,1>;
  template class WaveEquationOperationADERADCONFULL<3,1>;

  template class WaveEquationOperation<2,2>;
  template class WaveEquationOperation<3,2>;
  template class WaveEquationOperationADER<2,2>;
  template class WaveEquationOperationADER<3,2>;
  template class WaveEquationOperationADERLTS<2,2>;
  template class WaveEquationOperationADERLTS<3,2>;
  template class WaveEquationOperationADERADCONFULL<2,2>;
  template class WaveEquationOperationADERADCONFULL<3,2>;

  template class WaveEquationOperation<2,3>;
  template class WaveEquationOperation<3,3>;
  template class WaveEquationOperationADER<2,3>;
  template class WaveEquationOperationADER<3,3>;
  template class WaveEquationOperationADERLTS<2,3>;
  template class WaveEquationOperationADERLTS<3,3>;
  template class WaveEquationOperationADERADCONFULL<2,3>;
  template class WaveEquationOperationADERADCONFULL<3,3>;

  template class WaveEquationOperation<2,4>;
  template class WaveEquationOperation<3,4>;
  template class WaveEquationOperationADER<2,4>;
  template class WaveEquationOperationADER<3,4>;
  template class WaveEquationOperationADERLTS<2,4>;
  template class WaveEquationOperationADERLTS<3,4>;
  template class WaveEquationOperationADERADCONFULL<2,4>;
  template class WaveEquationOperationADERADCONFULL<3,4>;

  template class WaveEquationOperation<2,5>;
  template class WaveEquationOperation<3,5>;
  template class WaveEquationOperationADER<2,5>;
  template class WaveEquationOperationADER<3,5>;
  template class WaveEquationOperationADERLTS<2,5>;
  template class WaveEquationOperationADERLTS<3,5>;
  template class WaveEquationOperationADERADCONFULL<2,5>;
  template class WaveEquationOperationADERADCONFULL<3,5>;

  template class WaveEquationOperation<2,6>;
  template class WaveEquationOperation<3,6>;
  template class WaveEquationOperationADER<2,6>;
  template class WaveEquationOperationADER<3,6>;
  template class WaveEquationOperationADERLTS<2,6>;
  template class WaveEquationOperationADERLTS<3,6>;
  template class WaveEquationOperationADERADCONFULL<2,6>;
  template class WaveEquationOperationADERADCONFULL<3,6>;

  template class WaveEquationOperation<2,7>;
  template class WaveEquationOperation<3,7>;
  template class WaveEquationOperationADER<2,7>;
  template class WaveEquationOperationADER<3,7>;
  template class WaveEquationOperationADERLTS<2,7>;
  template class WaveEquationOperationADERLTS<3,7>;
  template class WaveEquationOperationADERADCONFULL<2,7>;
  template class WaveEquationOperationADERADCONFULL<3,7>;

  template class WaveEquationOperation<2,8>;
  template class WaveEquationOperation<3,8>;
  template class WaveEquationOperationADER<2,8>;
  template class WaveEquationOperationADER<3,8>;
  template class WaveEquationOperationADERLTS<2,8>;
  template class WaveEquationOperationADERLTS<3,8>;
  template class WaveEquationOperationADERADCONFULL<2,8>;
  template class WaveEquationOperationADERADCONFULL<3,8>;

  template class WaveEquationOperation<2,9>;
  template class WaveEquationOperation<3,9>;
  template class WaveEquationOperationADER<2,9>;
  template class WaveEquationOperationADER<3,9>;
  template class WaveEquationOperationADERLTS<2,9>;
  template class WaveEquationOperationADERLTS<3,9>;
  template class WaveEquationOperationADERADCONFULL<2,9>;
  template class WaveEquationOperationADERADCONFULL<3,9>;

  template class WaveEquationOperation<2,10>;
  template class WaveEquationOperation<3,10>;
  template class WaveEquationOperationADER<2,10>;
  template class WaveEquationOperationADER<3,10>;
  template class WaveEquationOperationADERLTS<2,10>;
  template class WaveEquationOperationADERLTS<3,10>;
  template class WaveEquationOperationADERADCONFULL<2,10>;
  template class WaveEquationOperationADERADCONFULL<3,10>;

  template class WaveEquationOperation<2,11>;
  template class WaveEquationOperation<3,11>;
  template class WaveEquationOperationADER<2,11>;
  template class WaveEquationOperationADER<3,11>;
  template class WaveEquationOperationADERLTS<2,11>;
  template class WaveEquationOperationADERLTS<3,11>;
  template class WaveEquationOperationADERADCONFULL<2,11>;
  template class WaveEquationOperationADERADCONFULL<3,11>;

  template class WaveEquationOperation<2,12>;
  template class WaveEquationOperation<3,12>;
  template class WaveEquationOperationADER<2,12>;
  template class WaveEquationOperationADER<3,12>;
  template class WaveEquationOperationADERLTS<2,12>;
  template class WaveEquationOperationADERLTS<3,12>;
  template class WaveEquationOperationADERADCONFULL<2,12>;
  template class WaveEquationOperationADERADCONFULL<3,12>;

}

