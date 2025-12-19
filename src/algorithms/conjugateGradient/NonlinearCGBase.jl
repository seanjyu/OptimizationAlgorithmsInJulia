"""
Interface Module for Nonlinear Conjugate Gradient methods
    Type: 
        NonlinearCGMethod - In concrete implementation assign type to struct and store hyperparameters
        
    Function(s):
        calculateBeta - Method interface to perform beta calculation to update direction based on specific Nonlinear Conjugate Gradient method
"""
abstract type NonlinearCGMethod end

"""
calculateBeta
    Method interface for generic Nonlinear Conjugate Gradient method
    Note not all concrete implementations require all inputs but they have been included for generalization purposes

Input
    gradNew (Vector)
    grad (Vector)
    direction (Vector)

Output - Vector with beta depending on specific Nonlinear Conjugate Gradient method
"""
function calculateBeta(nonlinearCGMethod::NonlinearCGMethod, gradNew, grad, direction)
    error("calculateBeta is not implemented for $(typeof(NonlinearCGMethod))")
end

"""
NonlinearCGOpt
    Implementation of generic Nonlinear Conjugate Gradient optimization loop

Reference(s)

Inputs
    f (function) - Objective function 
    x0 (Vector) - Starting coordinate        
    gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details
    nonlinearCGMethod (NonlinearCGMethod) - NonlinearCGMethod struct implementing QuasiNewtonInterface with a specific nonlinear conjugate gradient method
    lineSearchMethod (LineSearchMethod)- LineSearchMethod struct impementing lineSearchInterface with a specific step length algorithm 
    alpha (Float64) - Initial step length
    tol (Float64) - Stop criteria, if norm grad smaller than tolerance value iterations will stop
    lim (Int) - Maximum number of iterations
    lineSearchLim (Int) - Limit of number of line searching iterations 
    printIter (Bool) - Print number of iterations after Quasi-Newton method converges
    
Output - named tuple with the following fields
    minimum (Vector) - Final coordinate 
    path (Array) - Coordinates at each iteration 
    gradients (Array) - Gradient values at each iteration
    directions (Array) - Direction values at each iteration
    functionValues (Array) - Objective function values at each iteration
    iterations (Int) - Bumber of iterations 
"""

function NonlinearCGOpt(f, x0, gradEstimator::GradientEstimator, nonlinearCGMethod::NonlinearCGMethod, lineSearchMethod::LineSearchMethod,  criteria::ConvergenceCriteria; alpha = 1, tol = 1e-5, lim = 100, lineSearchLim = 100, printIter = false, track = true)
    
    x = copy(x0)
    
    fCur = f(x0)

    n = length(x0)

    gradRes = gradient(gradEstimator, f, x0)
    grad = gradRes.grad
    direction = -grad
    
    
    algorithmData = track ? 
        AlgorithmData(lim; directions=Vector{Float64}) : 
        NoAlgorithmData()

    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim
        xOld = copy(x)
        fOld = fCur
        
        # Perform line search 
        stepSearchResult = stepSearch(lineSearchMethod, gradEstimator, f, direction, x, fCur, grad, alpha, lineSearchLim)
        xNew = stepSearchResult.xNew
        
        fCur = f(xNew)
        
        gradNewRes = gradient(gradEstimator, f, xNew)
        gradNew = gradNewRes.grad

        # Restart direction every n iterations to stop accumulation of errors
        if mod(i, n) == 0
            direction = -gradNew  # Reset to steepest descent
        else
            beta = calculateBeta(nonlinearCGMethod, gradNew, grad, direction)
            direction = -gradNew + beta * direction
        end

        grad = gradNew

        # Log iteration data
        logIter!(logger, 
                fCur, 
                xNew, 
                gradNew, 
                norm(xNew - x),  # step size
                1 + gradRes.funcEvals + gradNewRes.funcEvals + stepSearchResult.funcEvals,  # func evals
                gradRes.funcEvals + gradNewRes.funcEvals,  # grad Est func evals
                1,  # grad evals
                directions=direction)

        # Update for next iteration 
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

    if logger.iterations[] == lim && isempty(logger.convergenceReason[])
        setConvergenceReason!(logger, "Maximum iterations reached")
    end

    finalizeLogger!(logger)
    
    return (
        minimum = x,
        finalValue = fCur,
        logger = logger
    )

end
