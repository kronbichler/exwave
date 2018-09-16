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

#include "../include/elementwise_cg.h"
#include "../include/elementwise_cg.templates.h"

namespace IterativeHelper
{

  template<typename value_type>
  SolverCGvect<value_type>::SolverCGvect(const unsigned int unknowns,
                                         const double abs_tol,
                                         const double rel_tol,
                                         const unsigned int max_iter):
    ABS_TOL(abs_tol),
    REL_TOL(rel_tol),
    MAX_ITER(max_iter),
    M(unknowns)
  {
    storage.resize(3*M);
    p = storage.begin();
    r = storage.begin()+M;
    v = storage.begin()+2*M;
  }

  template<typename value_type>
  value_type SolverCGvect<value_type>::l2_norm(const value_type *vector)
  {
    return std::sqrt(inner_product(vector, vector));
  }

  template<typename value_type>
  void SolverCGvect<value_type>::vector_init(value_type *vector)
  {
    for (unsigned int i=0; i<M; ++i)
      vector[i] = 0.0;
  }

  template<typename value_type>
  void SolverCGvect<value_type>::equ(value_type *dst, const value_type scalar, const value_type *in_vector)
  {
    for (unsigned int i=0; i<M; ++i)
      dst[i] = scalar*in_vector[i];
  }

  template<typename value_type>
  void SolverCGvect<value_type>::equ(value_type *dst, const value_type scalar1, const value_type *in_vector1, const value_type scalar2, const value_type *in_vector2)
  {
    for (unsigned int i=0; i<M; ++i)
      dst[i] = scalar1*in_vector1[i]+scalar2*in_vector2[i];
  }

  template<typename value_type>
  void SolverCGvect<value_type>::add(value_type *dst, const value_type scalar, const value_type *in_vector)
  {
    for (unsigned int i=0; i<M; ++i)
      dst[i] += scalar*in_vector[i];
  }

  template<typename value_type>
  value_type SolverCGvect<value_type>::inner_product(const value_type *vector1, const value_type *vector2)
  {
    value_type result = value_type();
    for (unsigned int i=0; i<M; ++i)
      result += vector1[i]*vector2[i];

    return result;
  }

  template class SolverCGvect<dealii::VectorizedArray<double>>;

}
