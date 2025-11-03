module QuasiNewtonInterface
    export QuasiNewtonMethod, updateApproximation

    """
    Interface Module for Quasi Newton methods
        Type: 
            QuasiNewtonMethod - In concrete implementation assign type to struct and store hyperparameters
        
        Required parameter
            inverseApproximation (bool) - Boolean value indicating whether this method estimates the hessian or inverse hessian
            
        Function(s):
            updateApproximation - Method interface to perform hessian or inverse hessian matrix update based on specific Quasi-Newton method
    """
    abstract type QuasiNewtonMethod end

    inverseApproximation(method::QuasiNewtonMethod) = method.inverseApproximation

    """
    updateApproximation
        Method interface for generic hessian/hessian inverse approximation function
        Note not all concrete implementations require all inputs but they have been included for generalization purposes
    Input
        quasiNewtonMethod (QuasiNewtonMethod) - Struct containing hyperparameters for the specific Quasi-Newton method
        M (Matrix) - Hessian/Inverse Hessian matrix to be updated
        s (Vector) - Difference in coordinate
        y (Vector) - Difference in function evaluation

    Output - Matrix with updated hessian/inverse hessian 
    """
    function updateApproximation(quasiNewtonMethod::QuasiNewtonMethod, M, s, y)
        error("updateApproximation not implemented for $(typeof(quasiNewtonMethod))")
    end
end

module QuasiNewtonBaseModule
    using ..QuasiNewtonInterface: QuasiNewtonMethod, updateApproximation, inverseApproximation
    using ..GradientEstimatorInterface: GradientEstimator, gradient
    using ..lineSearchInterface:LineSearchMethod, stepSearch
    using LinearAlgebra

    export QuasiNewtonOpt

    function QuasiNewtonOpt(f, x0, gradEstimator::GradientEstimator, quasiNewtonMethod::QuasiNewtonMethod, lineSearchMethod::LineSearchMethod; alpha = 1, beta = 1, tol = 1e-5, lim = 100, lineSearchLim = 100, printIter = false)
        """
        QuasiNewtonOpt
            Implementation of generic Quasi-Newton optimization loop

        Reference(s)
            This implementation generally follows algorithm 3.2 (pg 48) from Nocedal and Wright's 'Numerical Optimization' (2nd Ed).     
        
        Input:
            f (function) - Objective function 
            x0 (Vector) - Starting coordinate 
            gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details
            quasiNewtonMethod (QuasiNewtonMethod) - QuasiNewtonMethod struct implementing QuasiNewtonInterface with a specific Quasi-Newton method
            lineSearchMethod (LineSearchMethod)- LineSearchMethod struct impementing lineSearchInterface with a specific step length algorithm 
            alpha (Float64) - Initial step length
            beta (Float64) - Initial scalar for initial hessian/hessian inverse approximation (Initial estimate = beta * I)
            tol (Float64) - Stop criteria, if norm grad smaller than tolerance value iterations will stop
            lim (Int) - Maximum number of iterations
            lineSearchLim (Int) - Limit of number of line searching iterations 
            printIter (Bool) - Print number of iterations after Quasi-Newton method converges

        Output - named tuple with the following fields
            minimum (Vector) - Final coordinate 
            path (Array) - Coordinates at each iteration 
            gradients (Array) - Gradient values at each iteration
            hessians (Array) - Hessian values at each iteration
            functionValues (Array) - Objective function values at each iteration
            iterations (Int) - Bumber of iterations 
        """
        x = x0
        path = [copy(x0)]
        gradients = []
        hessians = []  
        functionValues = [f(x0)]

        n = length(x0)
        M = Matrix{Float64}(beta * I, n, n)  # Initialize as beta x identity
        grad = gradient(gradEstimator, f, x0)
        
        push!(gradients, copy(grad))
        push!(hessians, copy(M))  # Store initial approximation

        for i in 1:lim
            if norm(grad) < tol
                if printIter
                    println("Quasi Newton method converged in $i iterations. Final gradient norm: $(norm(grad))")
                end
                return (
                    minimum = x,
                    path = path,
                    gradients = gradients,
                    hessians = hessians,
                    functionValues = functionValues,
                    iterations = i
                )
            end

            # Compute search direction
            #TODO if B approximation is used then might want to use a different method to invert B matrix (Conjugate Gradient?)
            direction = inverseApproximation(quasiNewtonMethod) ? -M * grad : -(M \ grad)
            
            # Perform line search
            stepSearchResult = stepSearch(lineSearchMethod, gradEstimator, f, direction, x, f(x), grad, alpha, lineSearchLim)
            xNew = stepSearchResult.xNew

            push!(path, copy(xNew))
            push!(functionValues, f(xNew))

            # Compute new gradient
            gradNew = gradient(gradEstimator, f, xNew)
            
            # Update s and y
            s = xNew - x
            y = gradNew - grad
            M = updateApproximation(quasiNewtonMethod, M, s, y)
            
            # Store for next iteration
            push!(hessians, copy(M))
            push!(gradients, copy(gradNew))
            
            grad = gradNew
            x = xNew
        end
        
        error("Quasi-Newton Method failed: maximum iterations ($lim) reached")
    end

end 