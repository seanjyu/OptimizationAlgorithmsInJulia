module SR1Module
    using ..GradientEstimatorInterface: GradientEstimator, gradient, hessian
    using ..QuasiNewtonInterface: QuasiNewtonMethod
    import ..QuasiNewtonInterface: updateApproximation
    using LinearAlgebra
    export SR1H

    """
    Symmetric Rank 1 (SR1)
        Note the following module has 2 implementations, one for inverse hessian (BFGSH) one for hessian (BFGSB)

    Reference(s)
        See equation 6.24 (page 144) and equation 6.25 (page 144) in Nocedal and Wright's 'Numerical Optimization' (2nd Ed) for the Hessian approximation and
        inverse Hessian approximation respectively. 
    
    Hyperparameters
        inverseApproximation (Bool) - Boolean value representing whether the method is approximates the inverse hessian (true) or hessian matrix
        curvTol (Float64) - Tolerance for curvature condition
    """

    struct SR1H <: QuasiNewtonMethod 
        inverseApproximation::Bool
        tol::Float64
        SR1H(tol = 1e-8) = new(true, tol)
    end

    struct SR1B <: QuasiNewtonMethod 
        inverseApproximation::Bool
        curvTol::Float64
        SR1H(curvTol = 1e-8) = new(true, curvTol)
    end
    
    """
    updateApproximation 
        Concrete implementation of SR1 optimization
        This implementation makes use of Julia's multiple dispatch, that is the function that is executed depends on the type of QuasiNewtonMethod struct passed as input

    Input
        parameters (QuasiNewtonMethod) - Struct containing hyperparameters for the specific Quasi-Newton method
        M (Matrix) - Hessian/Inverse Hessian matrix to be updated
        s (Vector) - Difference in coordinate
        y (Vector) - Difference in function evaluation
    Output - Matrix with updated approximation
    """
    function updateApproximation(parameters::SR1H, H, s, y)
        # Compute the difference vector
        v = s - H * y
        
        # Compute the denominator
        denom = dot(v, y)
        
        # Check for numerical stability
        if abs(denom) < parameters.tol * norm(v) * norm(y)
            @warn "SR1H denominator too small, skipping update"
            return H
        end

        # Compute the SR1 update
        Hupdated = H + (v * v') / denom
        
        return Hupdated
    end

    function updateApproximation(parameters::SR1B, B, s, y)
        sr1Direction = y - B * s
        
        denominator = dot(sr1Direction, s)
        
        # Check SR1 condition (skip if denominator too small)
        if abs(denominator) <= parameters.sr1Tol
            @warn "SR1 denominator too small, skipping update"
            return B
        end
        
        # SR1 update
        return B + (sr1_direction * sr1_direction') / denominator
    end
end