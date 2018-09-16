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

#ifndef utilities_h_
#define utilities_h_

#include <deal.II/fe/mapping.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/function.h>

namespace HDG_WE
{
  using namespace dealii;


  struct Material
  {
    Material();

    double density;
    double speed;
  };


  class TimeControl
  {
  public:
    TimeControl();

    void setup(const double final_time_in,
               const double tick_time,
               const double time_step_in,
               const int    max_time_step_in = 0);

    void advance_time_step();

    void set_time_step(const double new_time_step);

    void set_time(const double new_time);

    double get_time() const;

    double get_time_step() const;

    unsigned int get_step_number () const;

    unsigned int get_output_step_number () const;

    bool done() const;

    // output is written if the time is slightly bigger than multiple of output_every_time
    bool at_tick() const;

  private:
    double final_time;
    double time_step;
    double time;
    double tick_size;
    unsigned int time_step_number;
    unsigned int max_time_step;
  };


  // Definition of an ExactSolution to specify pressure and velocity
  // components: dependent on the specified case, corresponding values are
  // returned
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution (const unsigned int n_components,
                   const unsigned int component,
                   const double time,
                   const int initial_cases,
                   const int membrane_modes) : Function<dim>(n_components, time),
      component(component),
      initial_cases(initial_cases),
      membrane_modes(membrane_modes)
    {}

    virtual double value (const Point<dim>   &p ,
                          const unsigned int  c=0) const;

  private:
    const int component;
    const int initial_cases;
    const int membrane_modes;
  };



  template <int dim>
  class ExactSolutionTimeDerivative : public Function<dim>
  {
  public:
    ExactSolutionTimeDerivative (const unsigned int n_components,
                                 const unsigned int component,
                                 const double time,
                                 const int initial_cases,
                                 const int membrane_modes) : Function<dim>(n_components, time),
      component(component),
      initial_cases(initial_cases),
      membrane_modes(membrane_modes)
    {}

    virtual double value (const Point<dim>   &p ,
                          const unsigned int  c=0) const;

  private:
    const int component;
    const int initial_cases;
    const int membrane_modes;
  };



  // this is a generic template for things to be done in each time step, such
  // as computing statistics on a surface integral. Since we do this by an
  // empty function here, so there is no need to define this function in case
  // nothing should be done. In case there is something to be done, one can
  // create a function with a more specialized vector type
  // (LinearAlgebra::distributed::Vector<Number>) that will be the first choice of
  // the compiler.
  template <int dim, typename VectorType>
  void time_step_analysis(const Mapping<dim> &,
                          const DoFHandler<dim> &,
                          const VectorType &,
                          const double           )
  {
  }

}

#endif
