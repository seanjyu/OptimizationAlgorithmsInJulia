module LBFGSModule
    using ..GradientEstimatorInterface: GradientEstimator, gradient, hessian
    using ..lineSearchInterface:LineSearchMethod, stepSearch
    using LinearAlgebra
    export LBFGS

    function LBFGS(f, x0, gradEstimator::GradientEstimator, lineSearchMethod::LineSearchMethod; m = 5, alpha = 1, beta = 1, tol = 1e-5, curvTol = 1e-8, lim = 100, lineSearchLim = 100, printIter = false)
        """
        Limited Broyden, Fletcher, Goldfarb, Shanno Optimization
            Implementation of LBFGS
        
        Reference(s)
            This implementation generally follows algorithm 7.4 (pg 178) from Nocedal and Wright's 'Numerical Optimization' (2nd Ed). 

        Input
            f (function) - Objective function 
            x0 (vector) - Starting coordinate 
            gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details
            lineSearchMethod (LineSearchMethod)- LineSearchMethod struct impementing lineSearchInterface with a specific step length algorithm 
            m (int) - Number of iterates used to approximate hessian matrix
            alpha (Float64) - Initial step length
            beta (Float64) - Initial scalar for initial hessian/hessian inverse approximation (Initial estimate = beta * I)
            tol (Float64) - Stop criteria, if norm grad smaller than tolerance value iterations will stop
            curvTol (Float64) - Tolerance for curvature requirement
            lim (Int) - Maximum number of iterations
            lineSearchLim (Int) - Limit of number of line searching iterations 
            printIter (Bool) - Print number of iterations after Quasi-Newton method converges

        Output - named tuple with the following fields
            minimum (Vector) - Final coordinate 
            path (Array) - Coordinates at each iteration 
            gradients (Array) - Gradient values at each iteration
            directions (Array) - Direction vectors at each iteration
            functionValues (Array) - Objective function values at each iteration
            iterations (Int) - Bumber of iterations 
        """
        
        x = x0
        path = [copy(x0)]
        gradients = []
        directions = []  
        functionValues = [f(x0)]
        s_history = []
        y_history = []

        n = length(x0)
        grad = gradient(gradEstimator, f, x0)
        
        push!(gradients, copy(grad))
        push!(directions, copy(-grad))  # for first step use negative gradient

        for i in 1:lim
            if norm(grad) < tol
                if printIter
                    println("Quasi Newton method converged in $i iterations. Final gradient norm: $(norm(grad))")
                end
                return (
                    minimum = x,
                    path = path,
                    gradients = gradients,
                    directions = directions,
                    functionValues = functionValues,
                    iterations = i
                )
            end
            
            if length(s_history) == 0
                direction = -beta * grad
            else
                lenLoop = min(length(s_history), m)
                a = zeros(m)
                q = copy(grad)
                # Compute search direction
                for j in lenLoop:-1:1
                    a[j] = s_history[j]' * q / (y_history[j]' * s_history[j])
                    q = q - a[j] * y_history[j]
                end
                
                p = (y_history[end]' * s_history[end]) / (y_history[end]' * y_history[end]) * q

                for j in 1:lenLoop
                    beta1 = (y_history[j]' * p) / (y_history[j]' * s_history[j])
                    p = p + (a[j] - beta1) * s_history[j] 
                end

                direction = -p

            end
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
            curvature = y' * s
            if curvature > curvTol
                push!(s_history, s)
                push!(y_history, y)

                if length(s_history) > m
                    popfirst!(s_history)
                    popfirst!(y_history)
                end
            else
                println("Skipping update - bad curvature")
            end
            
            # Store for next iteration
            push!(directions, copy(direction))
            push!(gradients, copy(gradNew))
            
            grad = gradNew
            x = xNew
        end
        
        # if maximum iterations reached throw error
        error("LBFGS Method failed: maximum iterations ($lim) reached")
    end



end