# ocotools
Create OCO problems and use them as benchmarks for OCO algorithms.

## Algorithm

An OCO algorithm. Is instantiated with the number of dimensions and (optional) hyper-parameters. Must implement the $\textttt{update}$ function which does its OCO update.

## Problem
An OCO problem. Gives access to the function value, gradient and Hessian at any point in its domain. Some have built-in update functions that represent the time-varying element of the problem.
