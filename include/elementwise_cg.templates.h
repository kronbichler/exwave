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

#ifndef elementwise_cg_template_h_
#define elementwise_cg_template_h_

#include "../include/elementwise_cg.h"

namespace IterativeHelper
{

  template <typename Number, typename Number2>
  bool all_smaller (const Number a, const Number2 b)
  {
    return a<b;
  }

  template <typename Number, typename Number2>
  bool all_smaller (const dealii::VectorizedArray<Number> a, const Number2 b)
  {
    for (unsigned int i=0; i<dealii::VectorizedArray<Number>::n_array_elements; ++i)
      if (a[i] >= b)
        return false;
    return true;
  }

  template <typename Number>
  void adjust_division_by_zero (Number &,
                                const double)
  {}

  template <typename Number>
  void adjust_division_by_zero (dealii::VectorizedArray<Number> &x,
                                const double tolerance)
  {
    // some components in a vectorized array might divide by zero which leads
    // to NaN values. To avoid that, we need to filter out too small
    // values. This has to be synchronized with the absolute tolerance in the
    // CG solver. The numbers that use this include a square of a norm, which
    // defines the upper bound. On the other hand, we cannot reach very small
    // tolerances if the number type does not allow for that (single precision
    // case)
    const double threshold = std::max(1e-5*tolerance*tolerance,
                                      1e4*static_cast<double>(std::numeric_limits<Number>::min()));
    for (unsigned int i=0; i<dealii::VectorizedArray<Number>::n_array_elements; ++i)
      if (x[i] < threshold)
        x[i] = 1;
  }



  template<typename value_type>
  template<typename Matrix>
  void SolverCGvect<value_type>::solve(const Matrix &matrix,
                                       value_type *solution,
                                       const value_type *rhs)
  {
    value_type one;
    one = 1.0;

    // guess initial solution
    vector_init(solution);

    // apply matrix vector product: v = A*solution
    matrix.vmult(v,solution);

    // compute residual: r = rhs-A*solution
    equ(r,one,rhs,-one,v);
    value_type norm_r0 = l2_norm(r);

    // precondition
    matrix.precondition(p,r);

    // compute norm of residual
    value_type norm_r_abs = norm_r0;
    value_type norm_r_rel = one;
    value_type r_times_y = inner_product(p, r);

    unsigned int n_iter = 0;

    while (true)
      {
        // v = A*p
        matrix.vmult(v,p);

        // p_times_v = p^T*v
        value_type p_times_v = inner_product(p,v);
        adjust_division_by_zero(p_times_v, ABS_TOL);

        // alpha = (r^T*y) / (p^T*v)
        value_type alpha = (r_times_y)/(p_times_v);

        // solution <- solution + alpha*p
        add(solution,alpha,p);

        // r <- r - alpha*v
        add(r,-alpha,v);

        // calculate residual norm
        norm_r_abs = l2_norm(r);
        norm_r_rel = norm_r_abs / norm_r0;

        // increment iteration counter
        ++n_iter;

        if (all_smaller(norm_r_abs, ABS_TOL) ||
            all_smaller(norm_r_rel, REL_TOL) || (n_iter > MAX_ITER))
          break;

        // precondition
        matrix.precondition(v,r);

        value_type r_times_y_new = inner_product(r,v);

        // beta = (v^T*r) / (p^T*v)
        value_type beta = r_times_y_new / r_times_y;

        // p <- r - beta*p
        equ(p,one,v,beta,p);

        r_times_y = r_times_y_new;
      }

    if (n_iter > MAX_ITER)
      {
        std::ostringstream message;
        for (unsigned int v=0; v<value_type::n_array_elements; v++)
          message << " v: " << v << "  " << norm_r_abs[v] << " ";
        Assert(n_iter <= MAX_ITER,
               dealii::ExcMessage("No convergence of solver in " + dealii::Utilities::to_string(MAX_ITER)
                                  + " iterations. Residual was " + message.str().c_str()));
      }
  }
}

#endif
