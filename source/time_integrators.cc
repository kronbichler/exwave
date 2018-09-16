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

#include <deal.II/lac/la_parallel_vector.h>

#include "../include/time_integrators.h"
#include "../include/wave_equation_operations.h"


DEAL_II_NAMESPACE_OPEN

template <typename VectorType, typename Operator>
void ExplicitEuler<VectorType,Operator>::perform_time_step(VectorType  &vec_n,
                                                           VectorType        &vec_np,
                                                           const double                    time_step,
                                                           Operator                       &op)
{
  op.apply(vec_n,vec_np);
  vec_np.sadd(time_step,1,vec_n);

  return;
}


template <typename VectorType, typename Operator>
void ArbitraryHighOrderDG<VectorType,Operator>::perform_time_step(VectorType    &vec_n,
    VectorType        &vec_np,
    const double                    ,
    Operator                       &op)
{
  // apply ader scheme
  op.apply_ader(vec_n,vec_np);

  // add
  vec_np.sadd(-1.0,1.0,vec_n);

  return;
}


template <typename VectorType, typename Operator>
void ArbitraryHighOrderDGLTS<VectorType,Operator>::perform_time_step(VectorType &vec_n,
    VectorType  &vec_np,
    const double              ,
    Operator                  &op)
{
  // apply ader lts scheme
  op.apply_ader(vec_n,vec_np);

  return;
}


template <typename VectorType, typename Operator>
void ClassRK4<VectorType,Operator>::perform_time_step(VectorType &vec_n,
                                                      VectorType       &vec_np,
                                                      const double                   time_step,
                                                      Operator                      &op)
{
  if (!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
      vec_tmp2.reinit(vec_np, true);
    }

  // stage 1
  op.apply(vec_n,vec_tmp1);
  vec_np = vec_n;
  vec_np.add(time_step/6., vec_tmp1);
  vec_tmp2 = vec_n;
  vec_tmp2.add(0.5*time_step, vec_tmp1);

  // stage 2
  op.apply(vec_tmp2,vec_tmp1);
  vec_np.add(time_step/3., vec_tmp1);
  vec_tmp2 = vec_n;
  vec_tmp2.add(0.5*time_step, vec_tmp1);

  // stage 3
  op.apply(vec_tmp2,vec_tmp1);
  vec_np.add(time_step/3., vec_tmp1);
  vec_tmp2 = vec_n;
  vec_tmp2.add(time_step, vec_tmp1);

  // stage 4
  op.apply(vec_tmp2,vec_tmp1);
  vec_np.add(time_step/6., vec_tmp1);

  return;
}


namespace
{
  template <typename Number>
  struct RKVectorUpdater
  {
    RKVectorUpdater (const Number  factor1,
                     const Number  factor2,
                     const bool    is_last,
                     LinearAlgebra::distributed::Vector<Number> &vector1,
                     LinearAlgebra::distributed::Vector<Number> &vector2,
                     LinearAlgebra::distributed::Vector<Number> &vector3)
      :
      factor1 (factor1),
      factor2 (factor2),
      is_last (is_last),
      vector1 (vector1),
      vector2 (vector2),
      vector3 (vector3)
    {
      AssertDimension(vector2.size(), vector1.size());
      AssertDimension(vector3.size(), vector1.size());
    }

    void
    apply_to_subrange (const std::size_t begin,
                       const std::size_t end) const
    {
      const Number factor1 = this->factor1;
      const Number factor2 = this->factor2;
      Number *vector1, *vector2, *vector3;
      vector1 = this->vector1.begin();
      vector2 = this->vector2.begin();
      vector3 = this->vector3.begin();
      if (is_last)
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i=begin; i<end; ++i)
            vector2[i] = vector3[i] + factor1 * vector1[i];

        }
      else
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i=begin; i<end; ++i)
            {
              const Number update = vector1[i];
              vector2[i] += factor1 * update;
              vector3[i] = vector2[i] + factor2 * update;
            }
        }
    }

    const Number factor1;
    const Number factor2;
    const bool   is_last;
    LinearAlgebra::distributed::Vector<Number> &vector1;
    LinearAlgebra::distributed::Vector<Number> &vector2;
    LinearAlgebra::distributed::Vector<Number> &vector3;
  };

  template<typename Number>
  struct RKVectorUpdatesRange : public parallel::ParallelForInteger
  {
    RKVectorUpdatesRange(const double  factor1,
                         const double  factor2,
                         const bool    is_last,
                         LinearAlgebra::distributed::Vector<Number> &vector1,
                         LinearAlgebra::distributed::Vector<Number> &vector2,
                         LinearAlgebra::distributed::Vector<Number> &vector3)
      :
      updater (factor1, factor2, is_last, vector1, vector2, vector3)
    {
      const std::size_t size = vector1.local_size();
      if (size < internal::VectorImplementation::minimum_parallel_grain_size)
        apply_to_subrange (0, size);
      else
        apply_parallel (0, size,
                        internal::VectorImplementation::minimum_parallel_grain_size);
    }

    ~RKVectorUpdatesRange() {}

    virtual void
    apply_to_subrange (const std::size_t begin,
                       const std::size_t end) const
    {
      updater.apply_to_subrange(begin, end);
    }

    const RKVectorUpdater<Number> updater;
  };
}



template <typename VectorType, typename Operator>
void LowStorageRK33Reg2<VectorType,Operator>::perform_time_step(VectorType &vec_n,
    VectorType &vec_np,
    const double             time_step,
    Operator                &op)
{
  const double a21 = 0.755726351946097;
  const double a32 = 0.386954477304099;

  const double b1 = 0.245170287303492;
  const double b2 = 0.184896052186740;
  const double b3 = 0.569933660509768;

  if (!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
    }

  typedef typename VectorType::value_type value_type;
  //stage 1
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a21*time_step, (b1-a21)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);
  // stage 2
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a32*time_step, (b2-a32)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);
  // stage 3
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(b3*time_step, 0, true,
                                   vec_tmp1, vec_np, vec_n);
}



template <typename VectorType, typename Operator>
void LowStorageRK45Reg2<VectorType,Operator>::perform_time_step(VectorType &vec_n,
    VectorType &vec_np,
    const double             time_step,
    Operator                 &op)
{
  typedef typename Operator::value_type value_type;

  const double a21 = 970286171893./4311952581923.;
  const double a32 = 6584761158862./12103376702013.;
  const double a43 = 2251764453980./15575788980749.;
  const double a54 = 26877169314380./34165994151039.;

  const double b1 =  1153189308089./22510343858157.;
  const double b2 = 1772645290293./ 4653164025191.;
  const double b3 = -1672844663538./ 4480602732383.;
  const double b4 =  2114624349019./3568978502595.;
  const double b5 =  5198255086312./14908931495163.;

  if (!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
    }

  // stage 1
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a21*time_step, (b1-a21)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);

  // stage 2
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a32*time_step, (b2-a32)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);

  // stage 3
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a43*time_step, (b3-a43)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);

  // stage 4
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a54*time_step, (b4-a54)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);

  // stage 5
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(b5*time_step, 0, true,
                                   vec_tmp1, vec_np, vec_n);
}


template <typename VectorType, typename Operator>
void LowStorageRK59Reg2<VectorType,Operator>::perform_time_step(VectorType &vec_n,
    VectorType &vec_np,
    const double             time_step,
    Operator                 &op)
{
  typedef typename Operator::value_type value_type;

  const double a21 = 1107026461565./5417078080134.;
  const double a32 = 38141181049399./41724347789894.;
  const double a43 = 493273079041./11940823631197.;
  const double a54 = 1851571280403./6147804934346.;
  const double a65 = 11782306865191./62590030070788.;
  const double a76 = 9452544825720./13648368537481.;
  const double a87 = 4435885630781./26285702406235.;
  const double a98 = 2357909744247./11371140753790.;

  const double b1 =  2274579626619./23610510767302.;
  const double b2 =  693987741272./12394497460941.;
  const double b3 =  -347131529483./15096185902911.;
  const double b4 =  1144057200723./32081666971178.;
  const double b5 =  1562491064753./11797114684756.;
  const double b6 =  13113619727965./44346030145118.;
  const double b7 =  393957816125./7825732611452.;
  const double b8 =  720647959663./6565743875477.;
  const double b9 =  3559252274877./14424734981077.;


  if (!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
    }

  // stage 1
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a21*time_step, (b1-a21)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);

  // stage 2
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a32*time_step, (b2-a32)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);

  // stage 3
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a43*time_step, (b3-a43)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);

  // stage 4
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a54*time_step, (b4-a54)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);

  // stage 5
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a65*time_step, (b5-a65)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);

  // stage 6
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a76*time_step, (b6-a76)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);

  // stage 7
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a87*time_step, (b7-a87)*time_step, false,
                                   vec_tmp1, vec_n, vec_np);

  // stage 8
  op.apply(vec_n, vec_tmp1);
  RKVectorUpdatesRange<value_type>(a98*time_step, (b8-a98)*time_step, false,
                                   vec_tmp1, vec_np, vec_n);

  // stage 9
  op.apply(vec_np, vec_tmp1);
  RKVectorUpdatesRange<value_type>(b9*time_step, 0, true,
                                   vec_tmp1, vec_np, vec_n);
}


template <typename VectorType, typename Operator>
void LowStorageRK45Reg3<VectorType,Operator>::perform_time_step(VectorType &vec_n,
    VectorType &vec_np,
    const double             time_step,
    Operator                &op)
{
  const double a21 = 2365592473904./8146167614645.;
  const double a32 = 4278267785271./6823155464066.;
  const double a43 = 2789585899612./8986505720531.;
  const double a54 = 15310836689591./24358012670437.;

  const double a31 = -722262345248./10870640012513.;
  const double a42 = 1365858020701./8494387045469.;
  const double a53 = 3819021186./2763618202291.;

  const double b1 = 846876320697./6523801458457.;
  const double b2 = 3032295699695./12397907741132.;
  const double b3 = 612618101729./6534652265123.;
  const double b4 = 1155491934595./2954287928812.;
  const double b5 = 707644755468./5028292464395.;

  if (!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np, true);
      vec_tmp2.reinit(vec_np, true);
    }

  // stage 1 -- begin
  op.apply(vec_n, vec_np);
  vec_n.add(a21*time_step,vec_np);
  vec_tmp1 = vec_n;
  vec_tmp1.add((b1-a21)*time_step,vec_np);
  // stage 1 -- end

  // stage 2 -- begin
  op.apply(vec_n, vec_tmp2);
  vec_tmp1.add(a32*time_step,vec_tmp2,(a31-b1)*time_step,vec_np);
  vec_np.sadd((b1-a31)*time_step,1,vec_tmp1);
  vec_np.add((b2-a32)*time_step,vec_tmp2);
  // stage 2 -- end

  // stage 3 -- begin
  op.apply(vec_tmp1,vec_n);
  vec_np.add(a43*time_step,vec_n,(a42-b2)*time_step,vec_tmp2);
  vec_tmp2.sadd((b2-a42)*time_step,1,vec_np);
  vec_tmp2.add((b3-a43)*time_step,vec_n);
  // stage 3 -- end

  // stage 4 -- begin
  op.apply(vec_np,vec_tmp1);
  vec_tmp2.add(a54*time_step,vec_tmp1,(a53-b3)*time_step,vec_n);
  vec_np = vec_tmp2;
  //vec_n.sadd((b3-a53)*time_step,1,vec_tmp2);
  vec_np.add((b3-a53)*time_step, vec_n, (b4-a54)*time_step,vec_tmp1);
  // stage 4 -- end

  // stage 5 -- begin
  op.apply(vec_tmp2,vec_tmp1);
  vec_np.add(b5*time_step,vec_tmp1);
  // stage 5 -- end

}

template <typename VectorType, typename Operator>
SSPRK<VectorType,Operator>::SSPRK(const unsigned int order,
                                  const unsigned int stages)
  :
  coeffs_are_initialized(false),
  order(order)
{
  initialize_coeffs(stages);
}


template <typename VectorType, typename Operator>
void SSPRK<VectorType,Operator>::perform_time_step (VectorType &vec_n,
                                                    VectorType &vec_np,
                                                    const double             time_step,
                                                    Operator                &op)
{
  const unsigned int stages = A.m();
  Assert(coeffs_are_initialized,ExcNotInitialized());

  // Initialize vectors if necessary
  if (vec_tmp1.empty() || !vec_tmp1[0].partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.resize(stages+1);
      for (unsigned int d=0; d<vec_tmp1.size(); ++d)
        vec_tmp1[d].reinit(vec_n, true);
      
      //vec_tmp2.reinit(vec_n, true);
      
      vec_tmp3.resize(stages+1);
      for (unsigned int d=0; d<vec_tmp3.size(); ++d)
        vec_tmp3[d].reinit(vec_n, true);
    }

  vec_tmp1[0] = vec_n;

  for (unsigned int s=1; s<stages+1; ++s)
  {
    op.apply(vec_tmp1[s-1], vec_tmp3[s-1]);      
    
    vec_tmp1[s] = 0.;
    for (unsigned int l=0; l<s; ++l)
      {
        vec_tmp1[s].add(A[s-1][l], vec_tmp1[l], B[s-1][l]*time_step, vec_tmp3[l]);
      }
  }

  vec_np = vec_tmp1[stages];
}



template <typename VectorType, typename Operator>
void SSPRK<VectorType,Operator>::initialize_coeffs (const unsigned int stages)
{
  A.reinit(stages, stages);
  B.reinit(stages, stages);
  if (order == 4)
    {
      if (stages == 5)
        {
          A[0][0] = 1.;
          A[0][1] = 0.;
          A[0][2] = 0.;
          A[0][3] = 0.;
          A[0][4] = 0.;
          A[1][0] = 0.261216512493821;
          A[1][1] = 0.738783487506179;
          A[1][2] = 0.;
          A[1][3] = 0.;
          A[1][4] = 0.;
          A[2][0] = 0.623613752757655;
          A[2][1] = 0.;
          A[2][2] = 0.376386247242345;
          A[2][3] = 0.;
          A[2][4] = 0.;
          A[3][0] = 0.444745181201454;
          A[3][1] = 0.120932584902288;
          A[3][2] = 0.;
          A[3][3] = 0.434322233896258;
          A[3][4] = 0.;
          A[4][0] = 0.213357715199957;
          A[4][1] = 0.209928473023448;
          A[4][2] = 0.063353148180384;
          A[4][3] = 0.;
          A[4][4] = 0.513360663596212;

          B[0][0] = 0.605491839566400;
          B[0][1] = 0.;
          B[0][2] = 0.;
          B[0][3] = 0.;
          B[0][4] = 0.;
          B[1][0] = 0.;
          B[1][1] = 0.447327372891397;
          B[1][2] = 0.;
          B[1][3] = 0.;
          B[1][4] = 0.;
          B[2][0] = 0.000000844149769;
          B[2][1] = 0.;
          B[2][2] = 0.227898801230261;
          B[2][3] = 0.;
          B[2][4] = 0.;
          B[3][0] = 0.002856233144485;
          B[3][1] = 0.073223693296006;
          B[3][2] = 0.;
          B[3][3] = 0.262978568366434;
          B[3][4] = 0.;
          B[4][0] = 0.002362549760441;
          B[4][1] = 0.127109977308333;
          B[4][2] = 0.038359814234063;
          B[4][3] = 0.;
          B[4][4] = 0.310835692561898;

          coeffs_are_initialized = true;
        }
      else if (stages == 6)
        {
          A[0][0] = 1.;
          A[0][1] = 0.;
          A[0][2] = 0.;
          A[0][3] = 0.;
          A[0][4] = 0.;
          A[0][5] = 0.;
          A[1][0] = 0.441581886978406;
          A[1][1] = 0.558418113021594;
          A[1][2] = 0.;
          A[1][3] = 0.;
          A[1][4] = 0.;
          A[1][5] = 0.;
          A[2][0] = 0.496140382330059;
          A[2][1] = 0.;
          A[2][2] = 0.503859617669941;
          A[2][3] = 0.;
          A[2][4] = 0.;
          A[2][5] = 0.;
          A[3][0] = 0.392013998230666;
          A[3][1] = 0.001687525300458;
          A[3][2] = -0.;
          A[3][3] = 0.606298476468875;
          A[3][4] = 0.;
          A[3][5] = 0.;
          A[4][0] = 0.016884674246355;
          A[4][1] = 0.000000050328214;
          A[4][2] = 0.000018549175549;
          A[4][3] = 0.;
          A[4][4] = 0.983096726249882;
          A[4][5] = 0.;
          A[5][0] = 0.128599802059752;
          A[5][1] = 0.150433518466544;
          A[5][2] = 0.179199506866483;
          A[5][3] = 0.173584325551242;
          A[5][4] = 0.;
          A[5][5] = 0.368182847055979;

          B[0][0] = 0.448860018455995;
          B[0][1] = 0.;
          B[0][2] = 0.;
          B[0][3] = 0.;
          B[0][4] = 0.;
          B[0][5] = 0.;
          B[1][0] = 0.;
          B[1][1] = 0.250651564517035;
          B[1][2] = 0.;
          B[1][3] = 0.;
          B[1][4] = 0.;
          B[1][5] = 0.;
          B[2][0] = 0.004050697317371;
          B[2][1] = 0.;
          B[2][2] = 0.226162437286560;
          B[2][3] = 0.;
          B[2][4] = 0.;
          B[2][5] = 0.;
          B[3][0] = 0.000000073512372;
          B[3][1] = 0.000757462637509;
          B[3][2] = -0.;
          B[3][3] = 0.272143145337661;
          B[3][4] = 0.;
          B[3][5] = 0.;
          B[4][0] = 0.000592927398846;
          B[4][1] = 0.000000022590323;
          B[4][2] = 0.000008325983279;
          B[4][3] = 0.;
          B[4][4] = 0.441272814688551;
          B[4][5] = 0.;
          B[5][0] = 0.000000009191468;
          B[5][1] = 0.067523591875293;
          B[5][2] = 0.080435493959395;
          B[5][3] = 0.077915063570602;
          B[5][4] = 0.;
          B[5][5] = 0.165262559524728;

          coeffs_are_initialized = true;
        }
      else if (stages == 7)
        {
          A[0][0] = 1.;
          A[0][1] = 0.;
          A[0][2] = 0.;
          A[0][3] = 0.;
          A[0][4] = 0.;
          A[0][5] = 0.;
          A[0][6] = 0.;
          A[1][0] = 0.277584603405600;
          A[1][1] = 0.722415396594400;
          A[1][2] = 0.;
          A[1][3] = 0.;
          A[1][4] = 0.;
          A[1][5] = 0.;
          A[1][6] = 0.;
          A[2][0] = 0.528403304637363;
          A[2][1] = 0.018109310473034;
          A[2][2] = 0.453487384889603;
          A[2][3] = 0.;
          A[2][4] = 0.;
          A[2][5] = 0.;
          A[2][6] = 0.;
          A[3][0] = 0.363822566916605;
          A[3][1] = 0.025636760093079;
          A[3][2] = 0.000072932527637;
          A[3][3] = 0.610467740462679;
          A[3][4] = 0.;
          A[3][5] = 0.;
          A[3][6] = 0.;
          A[4][0] = 0.080433061177282;
          A[4][1] = 0.000000001538366;
          A[4][2] = 0.000000000000020;
          A[4][3] = 0.000000000036824;
          A[4][4] = 0.919566937247508;
          A[4][5] = 0.;
          A[4][6] = 0.;
          A[5][0] = 0.305416318145737;
          A[5][1] = 0.017282647045059;
          A[5][2] = 0.214348299745317;
          A[5][3] = 0.001174022148498;
          A[5][4] = 0.003799138070873;
          A[5][5] = 0.457979574844515;
          A[5][6] = 0.;
          A[6][0] = 0.112741543203136;
          A[6][1] = 0.042888410429255;
          A[6][2] = 0.185108001868376;
          A[6][3] = 0.000003952121250;
          A[6][4] = 0.230275526732661;
          A[6][5] = 0.110240916986851;
          A[6][6] = 0.318741648658470;

          B[0][0] = 0.236998129331275;
          B[0][1] = 0.;
          B[0][2] = 0.;
          B[0][3] = 0.;
          B[0][4] = 0.;
          B[0][5] = 0.;
          B[0][6] = 0.;
          B[1][0] = 0.001205136607466;
          B[1][1] = 0.310012922173259;
          B[1][2] = 0.;
          B[1][3] = 0.;
          B[1][4] = 0.;
          B[1][5] = 0.;
          B[1][6] = 0.;
          B[2][0] = 0.000000000029361;
          B[2][1] = 0.007771318668946;
          B[2][2] = 0.194606801046999;
          B[2][3] = 0.;
          B[2][4] = 0.;
          B[2][5] = 0.;
          B[2][6] = 0.;
          B[3][0] = 0.001612059039346;
          B[3][1] = 0.011001602331536;
          B[3][2] = 0.000031297818569;
          B[3][3] = 0.261972390131100;
          B[3][4] = 0.;
          B[3][5] = 0.;
          B[3][6] = 0.;
          B[4][0] = 0.000000000027723;
          B[4][1] = 0.000000000660165;
          B[4][2] = 0.000000000000009;
          B[4][3] = 0.000000000015802;
          B[4][4] = 0.394617327778342;
          B[4][5] = 0.;
          B[4][6] = 0.;
          B[5][0] = 0.115125889382648;
          B[5][1] = 0.007416569384575;
          B[5][2] = 0.091984117559200;
          B[5][3] = 0.000503812679890;
          B[5][4] = 0.001630338861330;
          B[5][5] = 0.196534551952426;
          B[5][6] = 0.;
          B[6][0] = 0.000102167855778;
          B[6][1] = 0.018404869978158;
          B[6][2] = 0.079436115076445;
          B[6][3] = 0.000001695989127;
          B[6][4] = 0.098819030275264;
          B[6][5] = 0.047308112450629;
          B[6][6] = 0.136782840433305;

          coeffs_are_initialized = true;
        }
      else if (stages == 8)
        {
          A[0][0] = 1.;
          A[0][1] = 0.;
          A[0][2] = 0.;
          A[0][3] = 0.;
          A[0][4] = 0.;
          A[0][5] = 0.;
          A[0][6] = 0.;
          A[0][7] = 0.;
          A[1][0] = 0.538569155333175;
          A[1][1] = 0.461430844666825;
          A[1][2] = 0.;
          A[1][3] = 0.;
          A[1][4] = 0.;
          A[1][5] = 0.;
          A[1][6] = 0.;
          A[1][7] = 0.;
          A[2][0] = 0.004485387460763;
          A[2][1] = 0.;
          A[2][2] = 0.995514612539237;
          A[2][3] = 0.;
          A[2][4] = 0.;
          A[2][5] = 0.;
          A[2][6] = 0.;
          A[2][7] = 0.;
          A[3][0] = 0.164495299288580;
          A[3][1] = 0.016875060685979;
          A[3][2] = 0.;
          A[3][3] = 0.818629640025440;
          A[3][4] = 0.;
          A[3][5] = 0.;
          A[3][6] = 0.;
          A[3][7] = 0.;
          A[4][0] = 0.426933682982668;
          A[4][1] = 0.157047028197878;
          A[4][2] = 0.023164224070770;
          A[4][3] = 0.;
          A[4][4] = 0.392855064748685;
          A[4][5] = 0.;
          A[4][6] = 0.;
          A[4][7] = 0.;
          A[5][0] = 0.082083400476958;
          A[5][1] = 0.000000039091042;
          A[5][2] = 0.033974171137350;
          A[5][3] = 0.005505195713107;
          A[5][4] = 0.;
          A[5][5] = 0.878437193581543;
          A[5][6] = 0.;
          A[5][7] = 0.;
          A[6][0] = 0.006736365648625;
          A[6][1] = 0.010581829625529;
          A[6][2] = 0.009353386191951;
          A[6][3] = 0.101886062556838;
          A[6][4] = 0.000023428364930;
          A[6][5] = 0.;
          A[6][6] = 0.871418927612128;
          A[6][7] = 0.;
          A[7][0] = 0.071115287415749;
          A[7][1] = 0.018677648343953;
          A[7][2] = 0.007902408660034;
          A[7][3] = 0.319384027162348;
          A[7][4] = 0.007121989995845;
          A[7][5] = 0.001631615692736;
          A[7][6] = 0.;
          A[7][7] = 0.574167022729334;

          B[0][0] = 0.282318339066479;
          B[0][1] = 0.;
          B[0][2] = 0.;
          B[0][3] = 0.;
          B[0][4] = 0.;
          B[0][5] = 0.;
          B[0][6] = 0.;
          B[0][7] = 0.;
          B[1][0] = 0.;
          B[1][1] = 0.130270389660380;
          B[1][2] = 0.;
          B[1][3] = 0.;
          B[1][4] = 0.;
          B[1][5] = 0.;
          B[1][6] = 0.;
          B[1][7] = 0.;
          B[2][0] = 0.003963092203460;
          B[2][1] = 0.;
          B[2][2] = 0.281052031928487;
          B[2][3] = 0.;
          B[2][4] = 0.;
          B[2][5] = 0.;
          B[2][6] = 0.;
          B[2][7] = 0.;
          B[3][0] = 0.000038019518678;
          B[3][1] = 0.004764139104512;
          B[3][2] = 0.;
          B[3][3] = 0.231114160282572;
          B[3][4] = 0.;
          B[3][5] = 0.;
          B[3][6] = 0.;
          B[3][7] = 0.;
          B[4][0] = 0.000019921336144;
          B[4][1] = 0.044337256156151;
          B[4][2] = 0.006539685265423;
          B[4][3] = 0.;
          B[4][4] = 0.110910189373703;
          B[4][5] = 0.;
          B[4][6] = 0.;
          B[4][7] = 0.;
          B[5][0] = 0.000000034006679;
          B[5][1] = 0.000000011036118;
          B[5][2] = 0.009591531566657;
          B[5][3] = 0.001554217709960;
          B[5][4] = 0.;
          B[5][5] = 0.247998929466160;
          B[5][6] = 0.;
          B[5][7] = 0.;
          B[6][0] = 0.013159891155054;
          B[6][1] = 0.002987444564164;
          B[6][2] = 0.002640632454359;
          B[6][3] = 0.028764303955070;
          B[6][4] = 0.000006614257074;
          B[6][5] = 0.;
          B[6][6] = 0.246017544274548;
          B[6][7] = 0.;
          B[7][0] = 0.000000010647874;
          B[7][1] = 0.005273042658132;
          B[7][2] = 0.002230994887525;
          B[7][3] = 0.090167968072837;
          B[7][4] = 0.002010668386475;
          B[7][5] = 0.000460635032368;
          B[7][6] = 0.;
          B[7][7] = 0.162097880203691;

          coeffs_are_initialized = true;
        }
      else
        coeffs_are_initialized = false;
    }
  else
    coeffs_are_initialized = false;

  Assert(coeffs_are_initialized, ExcNotImplemented());
}

template class ExplicitEuler<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class ExplicitEuler<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class ArbitraryHighOrderDG<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class ArbitraryHighOrderDG<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class ArbitraryHighOrderDGLTS<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class ArbitraryHighOrderDGLTS<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class ClassRK4<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class ClassRK4<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class LowStorageRK33Reg2<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class LowStorageRK33Reg2<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class LowStorageRK45Reg2<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class LowStorageRK45Reg2<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class LowStorageRK45Reg3<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class LowStorageRK45Reg3<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class LowStorageRK59Reg2<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class LowStorageRK59Reg2<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;
template class SSPRK<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<2> >;
template class SSPRK<dealii::LinearAlgebra::distributed::Vector<double>, HDG_WE::WaveEquationOperationBase<3> >;

DEAL_II_NAMESPACE_CLOSE

