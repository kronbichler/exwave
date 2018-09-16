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

#ifndef time_integrators_h_
#define time_integrators_h_

#include <deal.II/algorithms/operator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/full_matrix.h>

DEAL_II_NAMESPACE_OPEN

// Base class for all time integrators
template <typename VectorType, typename Operator>
class ExplicitIntegrator
{
public:
  ExplicitIntegrator() {}

  ~ExplicitIntegrator() {}

  virtual void perform_time_step(VectorType  &vec_n,
                                 VectorType        &vec_np,
                                 const double                    time_step,
                                 Operator                       &op) = 0;

};

template <typename VectorType, typename Operator>
class ExplicitEuler : public ExplicitIntegrator<VectorType,Operator>
{
public:
  ExplicitEuler () {}

  virtual void perform_time_step(VectorType  &vec_n,
                                 VectorType        &vec_np,
                                 const double                    time_step,
                                 Operator                       &op);
};

template <typename VectorType, typename Operator>
class ArbitraryHighOrderDG : public ExplicitIntegrator<VectorType,Operator>
{
public:

  virtual void perform_time_step(VectorType    &vec_n,
                                 VectorType        &vec_np,
                                 const double                    ,
                                 Operator                       &op);
};

template <typename VectorType, typename Operator>
class ArbitraryHighOrderDGLTS : public ExplicitIntegrator<VectorType,Operator>
{
public:

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType  &vec_np,
                                 const double              ,
                                 Operator                  &op);
};

template <typename VectorType, typename Operator>
class ClassRK4 : public ExplicitIntegrator<VectorType,Operator>
{
public:
  ClassRK4 () {}

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType       &vec_np,
                                 const double                   time_step,
                                 Operator                      &op);

private:
  VectorType vec_tmp1, vec_tmp2;
};


template <typename VectorType, typename Operator>
class LowStorageRK33Reg2 : public ExplicitIntegrator<VectorType,Operator>
{
public:
  LowStorageRK33Reg2 () {}

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType &vec_np,
                                 const double             time_step,
                                 Operator                &op);

private:
  VectorType vec_tmp1;
};



template <typename VectorType, typename Operator>
class LowStorageRK45Reg2 : public ExplicitIntegrator<VectorType,Operator>
{
public:
  LowStorageRK45Reg2 () {}

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType &vec_np,
                                 const double             time_step,
                                 Operator                 &op);

private:
  double computing_times;
  VectorType vec_tmp1;
};

template <typename VectorType, typename Operator>
class LowStorageRK59Reg2 : public ExplicitIntegrator<VectorType,Operator>
{
public:
  LowStorageRK59Reg2 () {}

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType &vec_np,
                                 const double             time_step,
                                 Operator                 &op);

private:
  double computing_times;
  VectorType vec_tmp1;
};



template <typename VectorType, typename Operator>
class LowStorageRK45Reg3 : public ExplicitIntegrator<VectorType,Operator>
{
public:
  LowStorageRK45Reg3 () {}

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType &vec_np,
                                 const double             time_step,
                                 Operator                &op);

private:
  VectorType vec_tmp1, vec_tmp2;
};



template <typename VectorType, typename Operator>
class SSPRK : public ExplicitIntegrator<VectorType,Operator>
{
public:
  SSPRK(const unsigned int order,
        const unsigned int stages);

  virtual void perform_time_step(VectorType &vec_n,
                                 VectorType &vec_np,
                                 const double             time_step,
                                 Operator                &op);
private:
  FullMatrix<double> A,B;
  bool coeffs_are_initialized;
  void initialize_coeffs(const unsigned int stages);
  const unsigned int order;
  VectorType vec_tmp2;
  std::vector<VectorType> vec_tmp1, vec_tmp3;
};


DEAL_II_NAMESPACE_CLOSE

#endif
