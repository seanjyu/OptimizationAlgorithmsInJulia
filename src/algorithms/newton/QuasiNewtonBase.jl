
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
function QuasiNewtonOpt(f, x0, gradEstimator::GradientEstimator, quasiNewtonMethod::QuasiNewtonMethod, lineSearchMethod::LineSearchMethod, criteria::ConvergenceCriteria; alpha = 1, beta = 1, lim = 100, lineSearchLim = 100, printIter = false, track = true)
    
    x = x0
    fCur = f(x0)

    n = length(x0)
    M = Matrix{Float64}(beta * I, n, n)  # Initialize as beta x identity

    gradRes = gradient(gradEstimator, f, x0)
    grad = gradRes.grad

    algorithmData = track ? 
        AlgorithmData(lim; hessianApproxs=Matrix{Float64}) : 
        NoAlgorithmData()
    
    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim
        xOld = copy(x)
        fOld = fCur

        # Compute search direction
        #TODO if B approximation is used then might want to use a different method to invert B matrix (Conjugate Gradient?)
        direction = inverseApproximation(quasiNewtonMethod) ? -M * grad : -(M \ grad)
        
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

        # Update Hessian approximation
        M = updateApproximation(quasiNewtonMethod, M, s, y)

        # Log iteration data
        logIter!(logger, 
                fCur, 
                xNew, 
                gradNew, 
                norm(s),  # step size
                1 + gradRes.funcEvals + gradNewRes.funcEvals + stepSearchRes.funcEvals,  # func evals
                gradRes.funcEvals + gradNewRes.funcEvals,  # grad Est func evals
                stepSearchRes.gradEvals + 1,  # grad evals
                hessianApproxs=M)

        # Update for next iteration
        grad = gradNew
        gradRes = gradNewRes
        x = xNew

        # Check convergence
        converged, reason = CheckConvergence(criteria, gradNew, xNew, xOld, fCur, fOld, i)
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
            mininumPoint = x,
            finalValue = fCur,
            logger = logger
        )

end
    