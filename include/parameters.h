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

#ifndef parameters_h_
#define parameters_h_

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

enum class IntegratorType
{
  expleuler,     // 0 - explicit Euler
  classrk4,      // 1 - classical explicit Runge-Kutta with 4 stages
  lsrk45reg2,    // 2 - low storage Runge-Kutta
  lsrk33reg2,    // 3 - low storage Runge-Kutta
  lsrk45reg3,    // 4 - low storage Runge-Kutta
  lsrk59reg2,    // 5 - low starage Runge-Kutta
  ssprk,         // 6 - strong stability preserving Runge-Kutta
  ader,          // 7 - ADER time integration
  ader_lts,      // 8 - ADER with local time stepping
  ader_adconfull // 9 - ADER relying on the global derivative operator
};

class Parameters
{
public:
  Parameters ();

  void read_parameters (const std::string &parameter_filename);

  static void declare_parameters (ParameterHandler &prm);
  void parse_parameters (const std::string parameter_filename,
                         ParameterHandler &prm);

  void check_for_file (const std::string &parameter_filename,
                       ParameterHandler  &prm) const;

  void output_parameters (std::ostream &ostream);

  // use help
  ParameterHandler    prm;

  // geometry
  unsigned int        dimension;
  unsigned int        boundary_id;

  // spatial discretization
  unsigned int        fe_degree;
  unsigned int        n_refinements;
  unsigned int        n_adaptive_refinements;
  unsigned int        adaptive_refinement_interval;
  unsigned int        n_initial_intervals;
  double              grid_transform_factor;

  // temporal discretization
  IntegratorType      integ_type;
  double              cfl_number;
  unsigned int        max_time_steps;
  double              final_time;
  double              output_every_time;
  bool                write_vtu_output;
  bool                cfl_stability_analysis;

  // initial field
  unsigned int        initial_cases;
  unsigned int        membrane_modes;

  // ader specific
  bool                use_ader_post;
  bool                spectral_evaluation;

  // ader lts specific
  unsigned int        max_n_clusters;
  unsigned int        max_diff_clusters;

  // miscellaneous
  bool                output_of_parameters;
};



#endif
