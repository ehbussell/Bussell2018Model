// Function for the criterion of the problem
// Min criterion(z)

// The following are the input and output available variables 
// for the criterion of your optimal control problem.

// Input :
// dim_state : number of state variables
// initial_time : value of the initial time
// initial_state : initial state vector
// final_time : value of the final time
// final_state : final state vector
// dim_optimvars : number of optimization parameters
// optimvars : vector of optimization parameters
// dim_constants : number of constants
// constants : vector of constants

// Output :
// criterion : expression of the criterion to minimize

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states, controls, algebraic variables and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_criterion"
{
	// Objective function
	criterion = final_state[6];
}


