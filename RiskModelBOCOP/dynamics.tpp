// Function for the dynamics of the problem
// dy/dt = dynamics(y,u,z,p)

// The following are the input and output available variables 
// for the dynamics of your optimal control problem.

// Input :
// time : current time (t)
// normalized_time: t renormalized in [0,1]
// initial_time : time value on the first discretization point
// final_time : time value on the last discretization point
// dim_* is the dimension of next vector in the declaration
// state : vector of state variables
// control : vector of control variables
// algebraicvars : vector of algebraic variables
// optimvars : vector of optimization parameters
// constants : vector of constants

// Output :
// state_dynamics : vector giving the expression of the dynamic of each state variable.

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states, controls, algebraic variables and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_dynamics"
#include "adolc/adolc.h"
#include "adolc/adouble.h"
#include <iostream>
{
	// Implement risk based infection dynamics

	double birth = constants[0];
	double death = constants[1];
	double removal = constants[2];
	double recov = constants[3];
	double max_control_rate = constants[4];
	double beta [2][2] = {{constants[5], constants[6]},
						  {constants[7], constants[8]}};
	double sus_pow [2][2] = {{constants[9], constants[10]},
						  	 {constants[11], constants[12]}};
	double inf_pow [2][2] = {{constants[13], constants[14]},
						  	 {constants[15], constants[16]}};

	state_dynamics[6] = 0;

	for (int risk1=0; risk1<2; risk1++){
		state_dynamics[3*risk1] = birth * (state[3*risk1] + state[1+3*risk1] + state[2+3*risk1])
			- death * state[3*risk1] - max_control_rate * control[risk1] * state[3*risk1] / (
			state[3*risk1] + state[1+3*risk1] + state[2+3*risk1]) + recov * state[1+3*risk1];

		state_dynamics[1+3*risk1] = - state[1+3*risk1] * (death + removal + recov);

		state_dynamics[2+3*risk1] = - death * state[2+3*risk1] +
			max_control_rate * control[risk1] * state[3*risk1] / (
			state[3*risk1] + state[1+3*risk1] + state[2+3*risk1]);

		state_dynamics[6] += state[1+3*risk1];

		for (int risk2=0; risk2<2; risk2++){
			state_dynamics[3*risk1] -= beta[risk1][risk2]
				* pow(fabs(state[3*risk1]), sus_pow[risk1][risk2])
				* pow(fabs(state[1+3*risk2]), inf_pow[risk1][risk2]);
			
			state_dynamics[1+3*risk1] += beta[risk1][risk2]
				* pow(fabs(state[3*risk1]), sus_pow[risk1][risk2])
				* pow(fabs(state[1+3*risk2]), inf_pow[risk1][risk2]);
		}
	}

}
