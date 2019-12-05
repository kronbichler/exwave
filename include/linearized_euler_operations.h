// --------------------------------------------------------------------------
//
// Copyright (C) 2019 by the ExWave authors
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

#ifndef linearized_euler_operations_h_
#define linearized_euler_operations_h_

//#define GAUSS_POINTS_VECTOR_OPERATION

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include "parameters.h"
#include "utilities.h"

namespace DG_Euler
{
  using namespace dealii;

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
    AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim+2,Number> > phi;
    AlignedVector<VectorizedArray<Number> > coefficients;
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim+2,Number> inverse;
  };

  template<int dim>
  class LinearizedEulerOperationBase
  {
  public:
    typedef double value_type;

    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers) = 0;

    virtual std::string Name() = 0;

    // allow access to matrix free object
    virtual const MatrixFree<dim,value_type> &get_matrix_free() const = 0;

    // allow access to time control
    virtual HDG_WE::TimeControl &get_time_control() const = 0;

    // Standard evaluation routine
    virtual void apply (const LinearAlgebra::distributed::Vector<value_type> &src,
                        LinearAlgebra::distributed::Vector<value_type>       &dst) const = 0;

    // projection of initial field
    virtual void project_initial_field(LinearAlgebra::distributed::Vector<value_type> &solution,
                                       const Function<dim>                            &function) const = 0;

    virtual void estimate_error(const LinearAlgebra::distributed::Vector<value_type> &solution,
                                LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                                Vector<double>                                       &error_estimate) const = 0;
  };



  // Definition of the class LinearizedEulerOperation containing all evaluation
  // routines and some basic informations like time and material properties
  template<int dim, int fe_degree>
  class LinearizedEulerOperation : public LinearizedEulerOperationBase<dim>
  {
  public:
    typedef typename LinearizedEulerOperationBase<dim>::value_type value_type;
    static const int dimension = dim;

    // Constructor
    LinearizedEulerOperation(HDG_WE::TimeControl &time_control_in,
                             Parameters &parameters_in);

    // Destructor
    virtual ~LinearizedEulerOperation();

    virtual void setup(const Mapping<dim>                         &mapping,
                       const std::vector<const DoFHandler<dim> *> &dof_handlers);

    virtual std::string Name();

    // allow access to matrix free object
    const MatrixFree<dim,value_type> &get_matrix_free() const;

    // allow access to time control
    HDG_WE::TimeControl &get_time_control() const;

    // Standard evaluation routine
    void apply (const LinearAlgebra::distributed::Vector<value_type> &src,
                LinearAlgebra::distributed::Vector<value_type>       &dst) const;

    // projection of initial field
    void project_initial_field(LinearAlgebra::distributed::Vector<value_type> &solution,
                               const Function<dim>                                     &function) const;

    void estimate_error(const LinearAlgebra::distributed::Vector<value_type> &solution,
                        LinearAlgebra::distributed::Vector<value_type>       &tmp_vector,
                        Vector<double>                                                &error_estimate) const;

  protected:
    // matrix free object
    MatrixFree<dim,value_type>              data;

    // time control reference
    HDG_WE::TimeControl                             &time_control;

    // parameters reference
    Parameters                              &parameters;

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

    void local_apply_face (const MatrixFree<dim,value_type>                     &data,
                           LinearAlgebra::distributed::Vector<value_type>       &dst,
                           const LinearAlgebra::distributed::Vector<value_type> &src,
                           const std::pair<unsigned int,unsigned int>           &cell_range) const;

    void local_apply_boundary_face (const MatrixFree<dim,value_type>                     &data,
                                    LinearAlgebra::distributed::Vector<value_type>       &dst,
                                    const LinearAlgebra::distributed::Vector<value_type> &src,
                                    const std::pair<unsigned int,unsigned int>           &cell_range) const;

  };

}

#endif /* linearized_euler_operations_h_ */
