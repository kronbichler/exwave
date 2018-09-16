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

#include "../include/input_parameters.h"

namespace HDG_WE
{
  double grid_transform_factor = 0.0;

  std::vector<Material> input_materials()
  {
    // here, we only have one material...
    std::vector<Material> mats(1);
    mats[0].density = 1.0;
    mats[0].speed = 1.0;
    return mats;
  }

  void set_grid_transform_factor(double in)
  {
    grid_transform_factor = in;
    return;
  }

  double get_grid_transform_factor()
  {
    return grid_transform_factor;
  }
}
