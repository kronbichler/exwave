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

#include "../include/utilities.h"
#include "../include/input_parameters.h"

namespace HDG_WE
{

  Material::Material()
  {
    density = 1.0;
    speed   = 1.0;
  }

  TimeControl::TimeControl()
  {}

  void TimeControl::setup(const double final_time_in,
                          const double tick_time,
                          const double time_step_in,
                          const int    max_time_step_in)
  {
    final_time = final_time_in;
    time = 0.0;
    tick_size = tick_time;
    time_step_number = 0;
    time_step = time_step_in;
    time_step = (final_time-time)/std::max(std::round((final_time-time)/time_step),1.0);
    if (max_time_step_in==0)
      max_time_step = std::round((final_time-time)/time_step);
    else
      max_time_step = max_time_step_in;
  }

  void TimeControl::advance_time_step()
  {
    ++time_step_number;
    time += time_step;
  }

  void TimeControl::set_time_step(const double new_time_step)
  {
    time_step = new_time_step;
  }

  void TimeControl::set_time(const double new_time)
  {
    time = new_time;
  }

  double TimeControl::get_time() const
  {
    return time;
  }

  double TimeControl::get_time_step() const
  {
    return time_step;
  }

  unsigned int TimeControl::get_step_number () const
  {
    return time_step_number;
  }

  unsigned int TimeControl::get_output_step_number () const
  {
    return time/tick_size; // sync with at_tick
  }

  bool TimeControl::done() const
  {
    return time >= final_time || time_step_number>max_time_step;
  }

  bool TimeControl::at_tick() const
  {
    if (int(time/tick_size) != int((time-time_step)/tick_size) )
      return true;
    else
      return false;
  }


  template <int dim>
  double ExactSolution<dim>::value (const Point<dim>   &p,
                                    const unsigned int c) const
  {
    double t = this->get_time();
    double return_value = 0.;

    if (component<0)
      return_value = input_exact_solution(p,t,c,false,initial_cases,membrane_modes);
    else
      return_value = input_exact_solution(p,t,component,false,initial_cases,membrane_modes);

    return return_value;
  }


  template <int dim>
  double ExactSolutionTimeDerivative<dim>::value (const Point<dim>   &p,
                                                  const unsigned int c) const
  {
    double t = this->get_time();
    double return_value = 0.;

    if (component<0)
      return_value = input_exact_solution(p,t,c,true,initial_cases,membrane_modes);
    else
      return_value = input_exact_solution(p,t,component,true,initial_cases,membrane_modes);

    return return_value;
  }


  template class ExactSolution<2>;
  template class ExactSolution<3>;

}
