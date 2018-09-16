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

#ifndef elementwise_cg_h_
#define elementwise_cg_h_

#include <deal.II/base/vectorization.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/aligned_vector.h>

namespace IterativeHelper
{
  /**
   * Implementation of iterative conjugate gradient solver. The implementation
   * is inspired by the implementation of the CG method in deal.II but
   * specialized for the case of plain arrays and without classes. The reason
   * for this new implementation is that we want to solve several linear
   * systems at once with vectorized data types, which is the natural way of
   * doing things with the MatrixFree class.
   */
  template<typename value_type>
  class SolverCGvect
  {
  public:
    SolverCGvect(const unsigned int unknowns,
                 const double abs_tol=1.e-12,
                 const double rel_tol=1.e-8,
                 const unsigned int max_iter = 1e5);

    template <typename Matrix>
    void solve(const Matrix &matrix,  value_type *solution, const value_type *rhs);

  private:
    const double ABS_TOL;
    const double REL_TOL;
    const unsigned int MAX_ITER;
    dealii::AlignedVector<value_type> storage;
    value_type *p,*r,*v;
    const unsigned int M;
    value_type l2_norm(const value_type *vector);

    void vector_init(value_type *dst);
    void equ(value_type *dst, const value_type scalar, const value_type *in_vector);
    void equ(value_type *dst, const value_type scalar1, const value_type *in_vector1, const value_type scalar2, const value_type *in_vector2);
    void add(value_type *dst, const value_type scalar, const value_type *in_vector);
    value_type inner_product(const value_type *vector1, const value_type *vector2);
  };

}

#endif
