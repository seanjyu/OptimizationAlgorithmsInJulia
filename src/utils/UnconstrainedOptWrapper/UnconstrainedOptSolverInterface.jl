"""
Interface Module for Unconstrained Optimization Wrapper
    Type:
        UnconstrainedOptMethod - Struct to store hyperparameters
    Function(s):
        solveUnconstrainedOpt - Method that solves an unconstrained optimmization problem using method specified by the UnconstrainedOptMethod struct
"""
abstract type UnconstrainedOptMethod end

"""
solveUnconstrainedOpt
    Function to solve an unconstrained optimization problem using method specified by concrete UnconstrainedOptMethod instance

Input
    f (function) - Objective function 
    x0 (vector) - Starting coordinate
    parameters (UnconstrainedOptMethod) - UnconstrainedOptMethod struct containing parameters to the specific unconstrained optimization method

Output - Named tuple, fields depend on the specific method use, please see the relevant method's documentation 
"""
function solveUnconstrainedOpt(f, x0, parameters::UnconstrainedOptMethod)
    error("solveUnconstrainedOpt is not implemented for $(typeof(parameters))")
end
