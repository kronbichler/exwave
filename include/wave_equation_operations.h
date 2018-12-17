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

#ifndef wave_equation_operations_h_
#define wave_equation_operations_h_

//#define GAUSS_POINTS_VECTOR_OPERATION

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include "cluster_manager.h"
#include "cluster_manager.templates.h"
#include "elementwise_cg.h"
#include "elementwise_cg.templates.h"
#include "parameters.h"
#include "utilities.h"

namespace HDG_WE
{
  using namespace dealii;

  template<int, int> class WaveEquationOperationADER;
  template<int, int> class WaveEquationOperationADERLTS;
  template<int, int> class WaveEquationOperationADERADCONFULL;

  // Collect all data for the inverse mass matrix operation in a struct in
  // order to avoid allocating the memory repeatedly
  template <int dim, int fe_degree, typename Number>
  struct InverseMassMatrixData
  {
    InverseMassMatrixData(const MatrixFree<dim,Number> &data);

    // Manually implement the copy operator because CellwiseInverseMassMatrix
    // must point to the object 'phi'
    InverseMassMatrixData(const InverseMassMatrixData &other);

    // For memory alignment reasons, need to place the FEEvaluation object
    // into an aligned vector
    AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,Number> > phi;
    AlignedVector<VectorizedArray<Number> > coefficients;
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim+1,Number> inverse;
  };

  template<int dim>
  class WaveEquationOperationBase
  {
  public:
    typedef double value_type;
    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers,
                       const std::vector<Material>                &mats,
                       const std::vector<unsigned int>            &vectorization_categories = std::vector<unsigned int>()) = 0;

    virtual std::string Name() = 0;

    // allow access to matrix free object
    virtual const MatrixFree<dim,value_type> &get_matrix_free() const = 0;

    // allow access to time control
    virtual TimeControl &get_time_control() const = 0;

    // Standard evaluation routine
    virtual void apply (const LinearAlgebra::distributed::Vector<value_type> &src,
                        LinearAlgebra::distributed::Vector<value_type>       &dst) const = 0;

    // Standard evaluation routine
    virtual void apply_ader (const LinearAlgebra::distributed::Vector<value_type> &,
                             LinearAlgebra::distributed::Vector<value_type> &) const = 0;

    // projection of initial field
    virtual void project_initial_field(LinearAlgebra::distributed::Vector<value_type> &solution,
                                       const Function<dim>                                     &function) const = 0;

    virtual void compute_post_pressure(const LinearAlgebra::distributed::Vector<value_type> &solution,
                                       LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                                       LinearAlgebra::distributed::Vector<value_type>                     &post_pressure) const = 0;

    virtual void estimate_error(const LinearAlgebra::distributed::Vector<value_type> &solution,
                                LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                                Vector<double>                                                &error_estimate) const = 0;

    // return the cluster id and the time step (only interesting for ADER LTS)
    virtual unsigned int cluster_id(unsigned int ) const = 0;
    virtual value_type time_step(unsigned int ) const = 0;

  };



  // Definition of the class WaveEquationOperation containing all evaluation
  // routines and some basic informations like time and material properties
  template<int dim, int fe_degree>
  class WaveEquationOperation : public WaveEquationOperationBase<dim>
  {
  public:
    typedef typename WaveEquationOperationBase<dim>::value_type value_type;
    static const int dimension = dim;

    // Constructor
    WaveEquationOperation(TimeControl &time_control_in, Parameters &parameters_in);

    // Destructor
    virtual ~WaveEquationOperation();

    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers,
                       const std::vector<Material>                &mats,
                       const std::vector<unsigned int>            &vectorization_categories = std::vector<unsigned int>());

    virtual std::string Name();

    // Function to bring material parameters specified as input to the element
    // vectors
    void reset_data_vectors(const std::vector<Material> mats);

    // allow access to matrix free object
    const MatrixFree<dim,value_type> &get_matrix_free() const;

    // allow access to time control
    TimeControl &get_time_control() const;

    // Standard evaluation routine
    void apply (const LinearAlgebra::distributed::Vector<value_type> &src,
                LinearAlgebra::distributed::Vector<value_type>       &dst) const;

    // Standard evaluation routine
    virtual void apply_ader (const LinearAlgebra::distributed::Vector<value_type> &,
                             LinearAlgebra::distributed::Vector<value_type> &) const;

    // projection of initial field
    void project_initial_field(LinearAlgebra::distributed::Vector<value_type> &solution,
                               const Function<dim>                                     &function) const;

    void compute_post_pressure(const LinearAlgebra::distributed::Vector<value_type> &solution,
                               LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                               LinearAlgebra::distributed::Vector<value_type>                     &post_pressure) const;

    void estimate_error(const LinearAlgebra::distributed::Vector<value_type> &solution,
                        LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                        Vector<double>                                                &error_estimate) const;

    // return the cluster id (only interesting for ADER LTS)
    virtual unsigned int cluster_id(unsigned int ) const;

    virtual value_type time_step(unsigned int ) const;

    // return the speed of sound of a given element (cell and vect index required)
    value_type speed_of_sound(int cell_index, int vect_index) const;

  protected:
    // matrix free object
    MatrixFree<dim,value_type>              data;

    // time control reference
    TimeControl                             &time_control;

    // parameters reference
    Parameters                              &parameters;

    // Vectors with material properties for elements
    AlignedVector<VectorizedArray<value_type> > densities, speeds;

    // Mass matrix data
    //mutable std_cxx11::shared_ptr<InverseMassMatrixData<dim,fe_degree,value_type> > mass_matrix_data;
    mutable std::shared_ptr<InverseMassMatrixData<dim,fe_degree,value_type> > mass_matrix_data;

    // Vector to store computing times for different actions
    mutable std::vector<double>                    computing_times;

    void local_apply_mass_matrix(const MatrixFree<dim,value_type>                     &data,
                                 LinearAlgebra::distributed::Vector<value_type>       &dst,
                                 const LinearAlgebra::distributed::Vector<value_type> &src,
                                 const std::pair<unsigned int,unsigned int>           &cell_range) const;

    void local_apply_domain (const MatrixFree<dim,value_type>                     &data,
                             LinearAlgebra::distributed::Vector<value_type>       &dst,
                             const LinearAlgebra::distributed::Vector<value_type> &src,
                             const std::pair<unsigned int,unsigned int>           &cell_range) const;

    void evaluate_cell(FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type>  &phi_v,
                       FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type>    &phi_p,
                       const LinearAlgebra::distributed::Vector<value_type>    &src,
                       const unsigned int                                       cell) const;

    void local_apply_face (const MatrixFree<dim,value_type>                     &data,
                           LinearAlgebra::distributed::Vector<value_type>       &dst,
                           const LinearAlgebra::distributed::Vector<value_type> &src,
                           const std::pair<unsigned int,unsigned int>           &cell_range) const;

    void evaluate_inner_face(FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi,
                             FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_neighbor,
                             const LinearAlgebra::distributed::Vector<value_type>         &src,
                             const unsigned int                                            face,
                             const value_type                                              boundary_fac,
                             LinearAlgebra::distributed::Vector<value_type>               *dst) const;

    void local_apply_boundary_face (const MatrixFree<dim,value_type>                     &data,
                                    LinearAlgebra::distributed::Vector<value_type>       &dst,
                                    const LinearAlgebra::distributed::Vector<value_type> &src,
                                    const std::pair<unsigned int,unsigned int>           &cell_range) const;

    void evaluate_boundary_face(FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi,
                                const LinearAlgebra::distributed::Vector<value_type>         &src,
                                const unsigned int                                            face,
                                const value_type                                              boundary_fac,
                                LinearAlgebra::distributed::Vector<value_type>               *dst) const;


    // need this for local_apply_mass_matrix
    template <int, int> friend class WaveEquationOperationADER;
    template <int, int> friend class WaveEquationOperationADERADCONFULL;
  };

  template<int dim, int fe_degree>
  class WaveEquationOperationADER : public WaveEquationOperation<dim,fe_degree>
  {
  public:
    typedef typename WaveEquationOperation<dim,fe_degree>::value_type value_type;

    WaveEquationOperationADER(TimeControl &time_control_in, Parameters &parameters_in);

    virtual ~WaveEquationOperationADER();

    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers,
                       const std::vector<Material>                &mats,
                       const std::vector<unsigned int>            &vectorization_categories = std::vector<unsigned int>());

    virtual std::string Name();

    // Overwrite base evaluation routine
    virtual void apply_ader (const LinearAlgebra::distributed::Vector<value_type> &src,
                             LinearAlgebra::distributed::Vector<value_type>       &dst) const;

  protected:
    // additional vector to work with (stores temporal values between first and second evaluation)
    mutable LinearAlgebra::distributed::Vector<value_type> tempsrc;

    // temporary data structures for particular evaluation within
    // Cauchy-Kovalewski procedure
    std::vector<internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<value_type> > > shape_infos;
    std::vector<internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<value_type> > > shape_infos_embed;

    // for ADER, we need to different domain actions
    virtual void local_apply_firstader_domain (const MatrixFree<dim,value_type>                     &data,
                                               LinearAlgebra::distributed::Vector<value_type>       &dst,
                                               const LinearAlgebra::distributed::Vector<value_type> &src,
                                               const std::pair<unsigned int,unsigned int>           &cell_range) const;

    virtual void local_apply_secondader_domain (const MatrixFree<dim,value_type>                     &data,
                                                LinearAlgebra::distributed::Vector<value_type>       &dst,
                                                const LinearAlgebra::distributed::Vector<value_type> &src,
                                                const std::pair<unsigned int,unsigned int>           &cell_range) const;

    void integrate_taylor_cauchykovalewski(const unsigned int                                        cell,
                                           FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval,
                                           const LinearAlgebra::distributed::Vector<value_type>     &src,
                                           const value_type                                          t2,
                                           const value_type                                          t1,
                                           const value_type                                          te,
                                           const LinearAlgebra::distributed::Vector<value_type>     &recongraddiv) const;

    template <int step_no, bool add_into_contrib>
    void integrate_taylor_cauchykovalewski_step(const unsigned int                                        cell,
                                                FEEvaluation<dim,fe_degree,fe_degree+1,dim+1,value_type> &phi_eval,
                                                VectorizedArray<value_type>                              *spectral_array,
                                                const value_type                                          tstart,
                                                const value_type                                          tend,
                                                VectorizedArray<value_type>                              *contrib) const;

    // overwrite face routines
    virtual void local_apply_ader_face (const MatrixFree<dim,value_type>                     &data,
                                        LinearAlgebra::distributed::Vector<value_type>       &dst,
                                        const LinearAlgebra::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>           &cell_range) const;

    virtual void local_apply_ader_boundary_face (const MatrixFree<dim,value_type>                     &data,
                                                 LinearAlgebra::distributed::Vector<value_type>       &dst,
                                                 const LinearAlgebra::distributed::Vector<value_type> &src,
                                                 const std::pair<unsigned int,unsigned int>           &cell_range) const;
  };



  template<int dim, int fe_degree>
  class WaveEquationOperationADERADCONFULL : public WaveEquationOperation<dim,fe_degree>
  {
  public:
    typedef typename WaveEquationOperation<dim,fe_degree>::value_type value_type;

    WaveEquationOperationADERADCONFULL(TimeControl &time_control_in, Parameters &parameters_in);

    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers,
                       const std::vector<Material>                &mats,
                       const std::vector<unsigned int>            &vectorization_categories = std::vector<unsigned int>());

    virtual std::string Name();

    // Overwrite base evaluation routine
    virtual void apply_ader (const LinearAlgebra::distributed::Vector<value_type> &src,
                             LinearAlgebra::distributed::Vector<value_type>       &dst) const;

  protected:
    // additional vector to work with (stores temporal values between first and second evaluation)
    mutable LinearAlgebra::distributed::Vector<value_type> tempsrc;
  };



  template<int dim, int fe_degree>
  class WaveEquationOperationADERLTS : public WaveEquationOperationADER<dim,fe_degree>
  {
    template <typename> friend class ClusterManager;

  public:
    typedef typename WaveEquationOperation<dim,fe_degree>::value_type value_type;

    WaveEquationOperationADERLTS(TimeControl &time_control_in, Parameters &parameters_in);

    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers,
                       const std::vector<Material>                &mats,
                       const std::vector<unsigned int>            &vectorization_categories = std::vector<unsigned int>());

    virtual std::string Name();

    // Overwrite base evaluation routine
    virtual void apply_ader (const LinearAlgebra::distributed::Vector<value_type> &src,
                             LinearAlgebra::distributed::Vector<value_type>       &dst) const;

    unsigned int cluster_id(unsigned int cell) const;

    virtual value_type time_step(unsigned int cell) const;

    void communicate_flux_memory() const;

  private:

    // cluster manager
    ClusterManager<value_type> cluster_manager;

    // current time
    mutable double time;

    // flux memory vector
    mutable LinearAlgebra::distributed::Vector<value_type> flux_memory;

    // we need this frequently for index calculations
    static const unsigned int n_vect = VectorizedArray<value_type>::n_array_elements;

    void evaluate_cells_and_faces_first_ader(const LinearAlgebra::distributed::Vector<value_type>  &src,
                                             LinearAlgebra::distributed::Vector<value_type>        &dst) const;

    void evaluate_cells_second_ader(const LinearAlgebra::distributed::Vector<value_type>  &src,
                                    LinearAlgebra::distributed::Vector<value_type>        &dst) const;

    virtual void local_apply_firstader_domain (const MatrixFree<dim,value_type>                              &data,
                                               LinearAlgebra::distributed::Vector<value_type>        &dst,
                                               const LinearAlgebra::distributed::Vector<value_type>  &src,
                                               const std::pair<unsigned int,unsigned int>                &cell_range) const;

    virtual void local_apply_secondader_domain (const MatrixFree<dim,value_type>                              &data,
                                                LinearAlgebra::distributed::Vector<value_type>        &dst,
                                                const LinearAlgebra::distributed::Vector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>                &cell_range) const;

    void local_apply_dummy_domain (const MatrixFree<dim,value_type>                              &data,
                                   LinearAlgebra::distributed::Vector<value_type>        &dst,
                                   const LinearAlgebra::distributed::Vector<value_type>  &src,
                                   const std::pair<unsigned int,unsigned int>                &cell_range) const;

    // overwrite face routines
    virtual void local_apply_ader_face (const MatrixFree<dim,value_type>       &data,
                                        LinearAlgebra::distributed::Vector<value_type>         &dst,
                                        const LinearAlgebra::distributed::Vector<value_type>   &src,
                                        const std::pair<unsigned int,unsigned int>                 &cell_range) const;

    virtual void local_apply_ader_boundary_face (const MatrixFree<dim,value_type>                              &data,
                                                 LinearAlgebra::distributed::Vector<value_type>        &dst,
                                                 const LinearAlgebra::distributed::Vector<value_type>  &src,
                                                 const std::pair<unsigned int,unsigned int>              &cell_range) const;

    virtual void local_apply_postprocessing_domain (const MatrixFree<dim,value_type>                              &data,
                                                    LinearAlgebra::distributed::Vector<value_type>        &dst,
                                                    const LinearAlgebra::distributed::Vector<value_type>  &src,
                                                    const std::pair<unsigned int,unsigned int>                &cell_range) const;

    virtual void local_apply_postprocessing_face (const MatrixFree<dim,value_type>                              &data,
                                                  LinearAlgebra::distributed::Vector<value_type>        &dst,
                                                  const LinearAlgebra::distributed::Vector<value_type>  &src,
                                                  const std::pair<unsigned int,unsigned int>                &cell_range) const;

    virtual void local_apply_postprocessing_boundary_face (const MatrixFree<dim,value_type>                              &data,
                                                           LinearAlgebra::distributed::Vector<value_type>        &dst,
                                                           const LinearAlgebra::distributed::Vector<value_type>  &src,
                                                           const std::pair<unsigned int,unsigned int>                &cell_range) const;

    void local_apply_postprocessing_mass_matrix(const MatrixFree<dim,value_type>                             &data,
                                                LinearAlgebra::distributed::Vector<value_type>       &dst,
                                                const LinearAlgebra::distributed::Vector<value_type> &src,
                                                const std::pair<unsigned int,unsigned int>               &cell_range) const;


    void reconstruct_div_grad(const LinearAlgebra::distributed::Vector<value_type> &src,
                              LinearAlgebra::distributed::Vector<value_type>       &dst) const;

  };

}

#endif /* wave_equation_operations_h_ */
