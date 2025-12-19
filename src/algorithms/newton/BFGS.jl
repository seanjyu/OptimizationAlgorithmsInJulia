
"""
Broyden, Fletcher, Goldfarb, Shanno Optimization (BFGS)
    Note the following module has 2 implementations, one for inverse hessian (BFGSH) one for hessian (BFGSB)

Reference(s)
    See equation 6.19 (page 140) and equation 6.17 (page 140) in Nocedal and Wright's 'Numerical Optimization' for the Hessian approximation and
    inverse Hessian approximation respectively. 


Hyperparameters
    inverseApproximation (Bool) - Boolean value representing whether the method is approximates the inverse hessian (true) or hessian matrix
    curvTol (Float64) - Tolerance for curvature condition
"""

# Inverse Hessian BFGS
struct BFGSH <: QuasiNewtonMethod 
    inverseApproximation::Bool
    curvTol::Float64
    BFGSH(curvTol = 1e-10) = new(true, curvTol)
end

# Hessian BFGS
struct BFGSB <: QuasiNewtonMethod 
    inverseApproximation::Bool
    curvTol::Float64
    BFGSH(tol = 1e-10) = new(false, tol)
end

"""
updateApproximation 
    Concrete implementation of BFGS optimization
    This implementation makes use of Julia's multiple dispatch, that is the function that is executed depends on the type of QuasiNewtonMethod struct passed as input

Input
    parameters (QuasiNewtonMethod) - Struct containing hyperparameters for the specific Quasi-Newton method
    M (Matrix) - Hessian/Inverse Hessian matrix to be updated
    s (Vector) - Difference in coordinate
    y (Vector) - Difference in function evaluation
Output - Matrix with updated approximation
"""

# Inverse hessian approximation
function updateApproximation(parameters::BFGSH, H, s, y)
    curvature = dot(y, s)

    # Check curvature condition
    if curvature <= parameters.curvTol
        @warn "Curvature condition violated, skipping update"
        return H
    end
    
    rho = 1.0 / curvature
    
    n = length(s)
    Imat = Matrix{Float64}(I, n, n)
    
    # Compute the update
    term1 = Imat - rho * (s * y')
    term2 = Imat - rho * (y * s')
    term3 = rho * (s * s')
    
    return term1 * H * term2 + term3
end

# Hessian approximation
function updateApproximation(parameters::BFGSB, B, s, y)
    curvature = dot(s, y)
    
    # Check curvature condition
    if curvature <= parameters.curvTol
        @warn "Curvature condition violated, skipping update"
        return B
    end
    
    # Compute update
    term2 = (B * s * s' * B) / (s' * B * s)
    term3 = (y * y') / curvature
    
    return B - term2 + term3
end

