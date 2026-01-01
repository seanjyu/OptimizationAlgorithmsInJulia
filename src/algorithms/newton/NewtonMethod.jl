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
function NewtonMethod(f, x0, gradEstimator::GradientEstimator, criteria::ConvergenceCriteria, lim = 100, track = true)

    x = copy(x0)
    fCur = f(x0)

    hessianType = x0 isa Number ? Float64 : Matrix{Float64}

    algorithmData = track ? 
        AlgorithmData(lim; hessians=hessianType) : 
        NoAlgorithmData()
    
    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim
        gradRes = gradient(gradEstimator, f, x)
        grad = gradRes.grad
        hessRes = hessian(gradEstimator, f, x)
        hess = hessRes.hess

        xOld = copy(x)
        fOld = fCur
        
        step = hess \ (-grad) 

        x = x .+ step
        fCur = f(x)

        logIter!(logger, 
                fCur, 
                x, 
                grad, 
                norm(step), 
                1 + gradRes.funcEvals,  # func evals
                gradRes.funcEvals,  # grad Est func evals
                1 + hessRes.gradEvals,  # grad evals
                hessians=hess)

        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, i)
        if converged
            setConvergenceReason!(logger, reason)
            break
        end
    
    end
    
    if logger.iterations[] == lim && isempty(logger.convergenceReason[])
        setConvergenceReason!(logger, "Maximum iterations reached")
    end

    finalizeLogger!(logger)
        return (
            minimumPoint = x,
            finalValue = fCur,
            logger = logger
        )

end

