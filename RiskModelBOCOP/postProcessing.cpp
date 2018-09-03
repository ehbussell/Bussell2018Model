// THIS FILE IS OPTIONAL !

// The following are the input and output available variables
// in post-processing function.

// Input :
// dim_* is the dimension of next vector in the declaration
// timeSteps : vector of time steps
// timeStages : vector of time stages
// state : vector of state variables
// control : vector of control variables
// algebraicvars : vector of algebraic variables
// optimvars : vector of optimization parameters
// constants : vector of constants
// boundaryCondMultiplier : vector of boundary constraints multipliers
// pathConstrMultiplier   : vector of path constraints multipliers
// adjointState           : vector of adjoint state

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_postProcessing"
{
	// Your post-processing function HERE:
    return 0;
}

