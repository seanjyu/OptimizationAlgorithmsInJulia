module NewtonMethodModule

    using ..GradientEstimatorInterface: GradientEstimator, gradient, hessian
    using LinearAlgebra
    export NewtonMethod
    """
    Newton Method Module
        Implementation of Newton step optimization method. In this implementation the Hessian is solved using Julia's built in matrix inverse function.

    Reference(s)
        This implementation generally follows algorithm 3.2 (pg 48) from Nocedal and Wright's 'Numerical Optimization' (2nd Ed). 
        The algorithm in the book includes a line search for the given Newton direction but it is not implemented here.    

    Input:
        f (function) - Objective function 
        x0 (vector) - Starting coordinate 
        gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details
        tol (Float64) - Stop criteria, if norm grad smaller than tolerance value iterations will stop
        lim (Int) - Maximum number of iterations

    Output - named tuple with the following fields
        minimum (Vector)- Final coordinate 
        path (Array) - Coordinates at each iteration 
        gradients (Array) - Gradient values at each iteration
        hessians (Array) - Hessian values at each iteration
        functionValues (Array) - Objective function values at each iteration
        iterations (Int) - Bumber of iterations 
    """
    function NewtonMethod(f, x0, gradEstimator::GradientEstimator, tol = 1e-5, lim = 100)
        x = x0
        path = [copy(x0)]
        gradients = []
        hessians = []
        functionValues = [f(x0)]

        for i in 1:lim
            grad = gradient(gradEstimator, f, x)
            hess = hessian(gradEstimator, f, x)
            push!(gradients, copy(grad))
            push!(hessians, copy(hess))

            Δx = hess \ (-grad) # 

            x = x .+ Δx

            push!(path, copy(x))
            push!(functionValues, f(x))

            if norm(grad) < tol
                return (
                    minimum = x,
                    path = path,
                    gradients = gradients,
                    hessians = hessians,
                    functionValues = functionValues,
                    iterations = i
                )
            end
        end


        error("Basic Newton Method failed: maximum iterations ($lim) reached")
    end
end

