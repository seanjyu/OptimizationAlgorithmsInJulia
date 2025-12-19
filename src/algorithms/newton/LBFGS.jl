function LBFGS(f, x0, gradEstimator::GradientEstimator, lineSearchMethod::LineSearchMethod, criteria::ConvergenceCriteria; m = 5, alpha = 1, beta = 1, tol = 1e-5, curvTol = 1e-8, lim = 100, lineSearchLim = 100, printIter = false, track = true)
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
    
    x = copy(x0)
    fCur = f(x0)
    # path = [copy(x0)]
    # gradients = []
    # directions = []  
    # functionValues = [f(x0)]
    s_history = []
    y_history = []

    # n = length(x0)
    gradRes = gradient(gradEstimator, f, x0)
    grad = gradRes.grad

    # push!(gradients, copy(grad))
    # push!(directions, copy(-grad))  # for first step use negative gradient
    
    algorithmData = track ? 
        AlgorithmData(lim; directions=Vector{Float64}) : 
        NoAlgorithmData()

    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)
    
    for i in 1:lim
        xOld = copy(x)
        fOld = fCur
        
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
        stepSearchRes = stepSearch(lineSearchMethod, gradEstimator, f, direction, x, f(x), grad, alpha, lineSearchLim)
        xNew = stepSearchRes.xNew

        fCur = f(xNew)

        # Compute new gradient
        gradNewRes = gradient(gradEstimator, f, xNew)
        gradNew = gradNewRes.grad
        
        # Update s and y
        s = xNew - x
        y = gradNew - grad

        # check curvature condition before updating history
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

        # Log iteration data
        logIter!(logger, 
                fCur, 
                xNew, 
                gradNew, 
                norm(s),  # step size
                1 + gradRes.funcEvals + gradNewRes.funcEvals + stepSearchRes.funcEvals,  # func evals
                gradRes.funcEvals + gradNewRes.funcEvals,  # grad Est func evals
                stepSearchRes.gradEvals + 1,  # grad evals
                directions=direction)

        
        # # Store for next iteration
        # push!(directions, copy(direction))
        # push!(gradients, copy(gradNew))

        # Update for next iteration (BEFORE convergence check!)
        grad = gradNew
        gradRes = gradNewRes
        x = xNew
        
        # Check convergence
        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, i)
        if converged
            setConvergenceReason!(logger, reason)
            break
        end
        
    end
    
    # # if maximum iterations reached throw error
    # error("LBFGS Method failed: maximum iterations ($lim) reached")

    if logger.iterations[] == lim && isempty(logger.convergenceReason[])
        setConvergenceReason!(logger, "Maximum iterations reached")
    end

    finalizeLogger!(logger)

    if printIter
        if isempty(logger.convergenceReason[])
            println("LBFGS did not converge, maximum iterations reached")
        else
            println("LBFGS converged in $(i) iterations")
        end
    end
    
    return (
        minimum = x,
        finalValue = fCur,
        logger = logger
    )
end