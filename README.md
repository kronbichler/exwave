# ExWave: A high performance solver for the acoustic wave equation

ExWave is a high performance solver based on spatial discretization with the discontinous
Galerkin method and temporal discretization with explicit time integration by means of
Runge-Kutta or arbitrary derivative time integration. ExWave is based on the deal.II
finite element library, github.com/dealii/dealii, and makes use of advanced techniques
such as parallel adaptive mesh refinementand  fast integration based on sum factorization.

# Performance

The performance of ExWave is described in the publication arxiv.org/abs/1805.03981.
[DOI](https://arxiv.org/abs/1805.03981)
```
@article{skwk18,
title = {Efficient explicit time stepping of high order discontinous {G}alerkin schemes for waves},
author = {Schoeder, S. and Kormann, K. and Wall, W.A. and Kronbichler, M.},
journal = {arXiv preprint \url{http://arxiv.org/abs/1805.03981}},
volume = {v1},
year = {2018}
}
```

# Getting Started

### Configuration of deal.II

The deal.II library is a prerequisite for ExWave. In a first step, deal.II must be
downloaded and configured following the instructions supplied by the deal.II documentation.
It is important, however, to build deal.II with support of HDF5, P4EST, and MPI. Therefore,
the following command should be used for cmake:
```
cmake -DP4EST_DIR=/path/to/p4est -DDEAL_II_WITH_P4EST=ON -DDEAL_II_WITH_MPI=ON -D HDF5_DIR=/path/to/hdf5 -DCMAKE_INSTALL_PREFIX=/path/to/install path/to/source
```

### Configuration of ExWave

After setting up deal.II, ExWave is build in its directory by running
```
cmake -DDEAL_II_DIR=/path/to/dealii/build .
```
followed by
```
make
```
and one can choose to make the release or the debug version. Tests of the code are run by
executing
```
ctest
```

Simulations are then run by calling
```
./explicit_wave [optional_parameter_file.prm]
```
The provided source code includes a default parameter ﬁle named default_parameters.prm, from
which input is read in case no parameter ﬁle name is provided. While method and code are quite
general, we will explain it along with an academic example for which the parameter ﬁle is
provided in the current code version. The example is a vibrating membrane problem for which 
analytical solutions are available and implemented. On the d-dimensional cube, a vibrating 
membrane with several modes with homogeneous Dirichlet boundary conditions is simulated. This
example allows to observe convergence orders and to make performance measurements. To solve
more general acoustics problems, the code in the input_parameters.h file can be changed to 
allow a read in of an externally created mesh and to supply boudary condition specifications.

# General code structure

The main class of our program is the class WaveEquationProblem, which is defined and implemented 
in explicit_wave.cc. Its method run() executes the time loop. Main components are a time 
integrator derived from ExplicitIntegrator and a spatial operator derived from 
WaveEquationOperationBase. The time integrators execute the vector updates and call the spatial 
operator application. For arbitrary derivative time integration, spatial and temporal evaluation 
are strongly interlinked and the entire evaluation takes place in WaveEquationOpeationADER. The 
local time stepping requires a complex update call that is handled by a ClusterManager which in 
turn is called by WaveEquationOperationADERLTS.

The class WaveEquationOperation is templated on the dimension and the polynomial degree k of the 
problem. It relies heavily on the MatrixFree class of the deal.II library and uses the optimized
evaluation routines FEEvaluation and FEFaceEvaluation. Matrix-free operator evaluation allows
for a much higher performance compared to classical matrix-based schemes, which is due to a
higher arithmetic density. Also, fast integration techniques relying on sum factorization 
utilizing the tensor product structure of the shape functions are used. 

In the file time_integrators.h, not only the time integrators but also an optimized vector 
updater are implemented. The optimized vector updater RKVectorUpdater only requires two vector
reads per stage in contrast to a non-optimized version requiring five vector reads per 
Runge-Kutta stage.

# Literature 

The software design of ExWave is described in the following paper:
```
@article{schoeder19exwave,
author = {Schoeder, S. and Wall, W.A. and Kronbichler, M.},
title = {A high performance discontinuous {G}alerkin solver for the acoustic wave equation},
journal = {Software X},
volume = {9},
pages = {49-54},
year = {2019},
doi = {10.1016/j.softx.2019.01.001}
}
```
The following articles provide background information on explicit time integration for the
acoustic wave equation with Runge-Kutta or arbitrary derivative time integation as well as
for matrix-free methods with sum factorization techniques.
```
@article{schoeder18efficiency,
author = {Schoeder, S. and Kormann, K. and Wall, W.A. and Kronbichler, M.},
title = {Efficient explicit time stepping of high order discontinuous {G}alerkin schemes for waves},
journal = {SIAM Journal on Scientific Computing},
volume = {40},
number = 6
pages = {C803-C826},
year = {2018},
doi = {10.1137/18M1185399}
}
@article{schoeder18ader,
author = {Schoeder, S. and Kronbichler, M. and Wall, W.A.},
title = {Arbitrary High-Order Explicit Hybridizable Discontinuous
         {G}alerkin Methods for the Acoustic Wave Equation},
journal = {Journal of Scientific Computing},
volume = {76},
pages = {969-1007},
year = {2018},
doi = {10.1007/s10915-018-0649-2},
}
@article{kk17,
title = {Fast matrix-free evaluation of discontinuous {G}alerkin finite element operators},
author = {M. Kronbichler and K. Kormann},
journal = {arXiv preprint \url{http://arxiv.org/abs/1711.03590}},
volume = {v1},
year = {2017}
}
@article{kk12,
author = "Kronbichler, M. and Kormann, K.",
title  = "A generic interface for parallel cell-based finite element operator application",
journal = "Computers \& Fluids",
volume  = 63,
pages   = "135--147",
year    = 2012,
doi     = "10.1016/j.compfluid.2012.04.012"
}
@article{ksmw15,
author = {Kronbichler, M. and Schoeder, S. and M\"uller, C. and Wall, W.A.},
title = {Comparison of implicit and explicit hybridizable discontinuous {G}alerkin 
         methods for the acoustic wave equation},
journal = {International Journal of Numerical Methods in Engineering},
volume = {106},
number = {9},
pages = {712--739},
year = {2016},
doi = {10.1002/nme.5137},
issn = {1097-0207}
}
```
