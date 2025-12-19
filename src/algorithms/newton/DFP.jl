

"""
D (DFP)
    Note the following module has 2 implementations, one for inverse hessian (DFPH) one for hessian (DFPB)

Reference(s)
    See equation 6.13 (page 139) and equation 6.15 (page 139) in Nocedal and Wright's 'Numerical Optimization' (2nd Ed) for the Hessian approximation and
    inverse Hessian approximation respectively. 

Hyperparameters
    inverseApproximation (Bool) - Boolean value representing whether the method is approximates the inverse hessian (true) or hessian matrix
    curvTol (Float64) - Tolerance for curvature condition
"""


struct DFPH <: QuasiNewtonMethod
    inverseApproximation::Bool
    curvTol::Float64
    DFPH(curvTol = 1e-8) = new(true, curvTol)
end

struct DFPB <: QuasiNewtonMethod
    inverseApproximation::Bool
    curvTol::Float64
    DFPH(curvTol = 1e-8) = new(true, curvTol)
end

"""
updateApproximation 
    Concrete implementation of DFP optimization
    This implementation makes use of Julia's multiple dispatch, that is the function that is executed depends on the type of QuasiNewtonMethod struct passed as an input

Input
    parameters (QuasiNewtonMethod) - Struct containing hyperparameters for the specific Quasi-Newton method
    M (Matrix) - Hessian/Inverse Hessian matrix to be updated
    s (Vector) - Difference in coordinate
    y (Vector) - Difference in function evaluation
Output - Matrix with updated approximation
"""

# Inverse hessian approximation
function updateApproximation(parameters::DFPH, H, s, y)
    curvature = dot(y, s)
    # Check curvature condition
    if curvature <= parameters.curvTol
        @warn "Curvature condition violated, skipping update"
        return H
    end

    term2 = (H * y * y' * H) / (y' * H * y) 
    term3 = (s * s') / curvature
    return H - term2 + term3
end

# Hessian approximation
function updateApproximation(parameters::DFPB, B, s, y)
    # Check curvature condition
    curvature = dot(y, s) 
    if curvature <= parameters.curvTol
        @warn "Curvature condition violated, skipping update"
        return B
    end
    term2 = (B * s * s' * B) / (s' * B * s)
    term3 = (y * y') / curvature
    
    return B - term2 + term3
end
