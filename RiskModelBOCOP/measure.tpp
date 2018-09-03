// THIS FILE IS OPTIONAL !

// The following are the input and output available variables
// for the measures of your optimal control problem.

// Input :
// k : number of the observation file which the measure refers to
// dim_time_observation : number of the observation times
// time : current time (t)
// state : vector of state variables
// constants : vector of constants
// optimvars : vector of optimization parameters 
// observations : vector of the observations for file k, at current time
// dim_* : dimension of vectors (dim_state is the dimension of state vector)

// Output :
// measures : vector giving the theoretical measure, this vector has the
// same dimension of the vector of observations

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_measure"
{
	// Your measure functions HERE:
	//measures[0] = ...;
}

