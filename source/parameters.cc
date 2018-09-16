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

#include <fstream>
#include "../include/parameters.h"


Parameters::Parameters()
  :
  dimension(numbers::invalid_unsigned_int)
{
  // do nothing
}


void Parameters::read_parameters (const std::string &parameter_filename)
{
  Parameters::declare_parameters (prm);
  check_for_file(parameter_filename, prm);
  parse_parameters (parameter_filename, prm);
}


void Parameters::declare_parameters (ParameterHandler &prm)
{
  prm.enter_subsection ("General");
  prm.declare_entry ("dimension","2",Patterns::Integer(),
                     "Defines the dimension of the problem.");
  prm.declare_entry ("boundary_id","2",Patterns::Integer(),
                     "Boundary condition id.");
  prm.declare_entry ("grid_transform_factor","0.1",Patterns::Double(),
                     "Distortion of mesh.");
  prm.declare_entry ("fe_degree","2",Patterns::Integer(),
                     "Polynomial degree of the shape functions.");
  prm.declare_entry ("n_initial_intervals","5",Patterns::Integer(),
                     "Number of elements on one edge of the base mesh.");
  prm.declare_entry ("n_refinements","0",Patterns::Integer(),
                     "Number of refinements of the base mesh.");
  prm.declare_entry ("n_adaptive_refinements","0",Patterns::Integer(),
                     "Number of adaptive refinements in h-adaptivity.");
  prm.declare_entry ("adaptive_refinement_interval","0",Patterns::Integer(),
                     "Steps for adaptivity update.");
  prm.leave_subsection();

  prm.enter_subsection ("TimeDiscretization");
  prm.declare_entry ("time_integrator","ADER",Patterns::Selection("ExplEuler|clRK4|LSRK45R2|LSRK33R2|LSRK45R3|LSRK59R2|SSPRK|ADER|ADERLTS|ADERADCONFULL"),
                     "Type of time integrator.");
  prm.declare_entry ("cfl_number","0.1",Patterns::Double(),
                     "Courant number.");
  prm.declare_entry ("max_time_steps","0",Patterns::Integer(),
                     "Maximal number of time steps.");
  prm.declare_entry ("final_time","1.0",Patterns::Double(),
                     "Maximal time.");
  prm.declare_entry ("output_every_time","0.1",Patterns::Double(),
                     "Output time step.");
  prm.declare_entry ("cfl_stability_analysis","false",Patterns::Bool(),
                     "Run of a time step stability analysis.");
  prm.leave_subsection();

  prm.enter_subsection ("InitialField");
  prm.declare_entry ("initital_cases","1",Patterns::Integer(),
                     "Initial case.");
  prm.declare_entry ("membrane_modes","2",Patterns::Integer(),
                     "Membrane modes in analytic solution.");
  prm.leave_subsection();

  prm.enter_subsection ("ADER_TimeIntegration");
  prm.declare_entry ("use_ader_post","false",Patterns::Bool(),
                     "Use of ADER Reconstruction for superconvergence.");
  prm.declare_entry ("spectral_evaluation","true",Patterns::Bool(),
                     "Spectral evaluation of Taylor-Cauchy-Kowalevski procedure.");
  prm.leave_subsection();

  prm.enter_subsection ("ADERLTS_TimeIntegration");
  prm.declare_entry ("max_n_clusters","10",Patterns::Integer(),
                     "Number of allowed time step clusters.");
  prm.declare_entry ("max_diff_clusters","7",Patterns::Integer(),
                     "Allowed time step difference between clusters.");
  prm.leave_subsection();

  prm.enter_subsection ("Miscellaneous");
  prm.declare_entry ("output_parameters","true",Patterns::Bool(),
                     "Output all used parameters in the end of the simulation.");
  prm.leave_subsection();
}


void Parameters::check_for_file (const std::string &parameter_filename,
                                 ParameterHandler  & /*prm*/) const
{
  std::ifstream parameter_file (parameter_filename.c_str());

  if (!parameter_file)
    {
      parameter_file.close ();

      std::ostringstream message;
      message << "Input parameter file <" << parameter_filename
              << "> not found. Please make sure the file exists!"
              << std::endl;

      AssertThrow (false, ExcMessage (message.str().c_str()));
    }
}


void Parameters::parse_parameters (const std::string parameter_file,
                                   ParameterHandler &prm)
{
  try
    {
      prm.parse_input (parameter_file);
    }
  catch (...)
    {
      AssertThrow (false, ExcMessage ("Invalid input parameter file."));
    }

  prm.enter_subsection("General");

  dimension = prm.get_integer ("dimension");
  boundary_id = prm.get_integer ("boundary_id");

  fe_degree = prm.get_integer ("fe_degree");
  n_refinements = prm.get_integer ("n_refinements");
  n_adaptive_refinements = prm.get_integer ("n_adaptive_refinements");
  adaptive_refinement_interval = prm.get_integer ("adaptive_refinement_interval");
  n_initial_intervals = prm.get_integer ("n_initial_intervals");
  grid_transform_factor = prm.get_double ("grid_transform_factor");

  prm.leave_subsection ();
  prm.enter_subsection("TimeDiscretization");

  std::string timestring = prm.get("time_integrator");
  if (timestring=="ExplEuler")
    {
      integ_type = IntegratorType::expleuler;
    }
  else if (timestring=="clRK4")
    {
      integ_type = IntegratorType::classrk4;
    }
  else if (timestring=="LSRK45R2")
    {
      integ_type = IntegratorType::lsrk45reg2;
    }
  else if (timestring=="LSRK33R2")
    {
      integ_type = IntegratorType::lsrk33reg2;
    }
  else if (timestring=="LSRK45R3")
    {
      integ_type = IntegratorType::lsrk45reg3;
    }
  else if (timestring=="LSRK59R2")
    {
      integ_type = IntegratorType::lsrk59reg2;
    }
  else if (timestring=="SSPRK")
    {
      integ_type = IntegratorType::ssprk;
    }
  else if (timestring=="ADER")
    {
      integ_type = IntegratorType::ader;
    }
  else if (timestring=="ADERLTS")
    {
      integ_type = IntegratorType::ader_lts;
    }
  else if (timestring=="ADERADCONFULL")
    {
      integ_type = IntegratorType::ader_adconfull;
    }
  else
    AssertThrow(false,
                ExcMessage("unknown time integrator " + timestring + " requested"));

  cfl_number = prm.get_double("cfl_number")/std::pow(fe_degree,1.5);
  max_time_steps = prm.get_integer("max_time_steps");
  final_time = prm.get_double("final_time");
  output_every_time = prm.get_double("output_every_time");
  cfl_stability_analysis = prm.get_bool("cfl_stability_analysis");

  prm.leave_subsection();
  prm.enter_subsection ("InitialField");

  initial_cases = prm.get_integer ("initital_cases");
  membrane_modes = prm.get_integer ("membrane_modes");

  prm.leave_subsection();
  prm.enter_subsection ("ADER_TimeIntegration");

  use_ader_post = prm.get_bool ("use_ader_post");
  spectral_evaluation  = prm.get_bool ("spectral_evaluation");

  prm.leave_subsection();
  prm.enter_subsection ("ADERLTS_TimeIntegration");

  max_n_clusters = prm.get_integer ("max_n_clusters");
  max_diff_clusters = prm.get_integer ("max_diff_clusters");

  prm.leave_subsection();
  prm.enter_subsection ("Miscellaneous");

  output_of_parameters = prm.get_bool ("output_parameters");

  prm.leave_subsection();
}


void Parameters::output_parameters (std::ostream &ostream)
{
  if (output_of_parameters)
    {
      ostream<<std::endl;
      ostream<< "The simulation was run with the following parameters: "<<std::endl;
      prm.print_parameters(ostream,prm.ShortText);
      ostream<<std::endl;
    }
}





