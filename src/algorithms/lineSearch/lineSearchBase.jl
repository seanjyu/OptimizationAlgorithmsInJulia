"""
Interface Module for Line Search methods
    Type: 
        LineSearchMethod - In concrete implementation assign type to struct and store hyperparameters
        
    Function(s):
        stepSearch - Method interface to perform step search based on specific line search method
"""
abstract type LineSearchMethod end

"""
stepSearch
    Method interface for generic line search method.
    Note not all concrete implementations require all inputs but they have been included for generalization purposes.
Input
    lineSearchMethod (LineSearchMethod) - Struct containing hyperparameters for the specific line search method 
    gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details 
    f (function) - Objective function to be optimized 
    direction (Vector) - current step direction 
    xCur (Vector) - Current coordinate
    fCur (Number) - Objective function evaluation at current coordinate 
    gradCur (Vector) - Current gradient 
    alpha (Float64) - Initial step length 
    lineSearchLim (Int) - Iteration limit for line search 

Output - named tuple with the following fields
    xNew (Vector) - New coordinate 
    fFinal (Vector) - Objective function evaluation at new coordinate 
    alphaFinal (Float64) - Final step size 
"""
function stepSearch(lineSearchMethod::LineSearchMethod, 
                    gradEstimator::GradientEstimator, 
                    f, 
                    direction, 
                    xCur, 
                    fCur,
                    gradCur, 
                    alpha,  
                    lineSearchLim)
    error("step search not implemented for $(typeof(lineSearchMethod))")
end


function lineSearch(f, x0, gradEstimator::GradientEstimator, lineSearchMethod::LineSearchMethod, criteria::ConvergenceCriteria; alpha = 1, lim = 100, lineSearchLim = 100, printIter = false, track = true)
    """
    lineSearch
        Implementation of generic line search

    Input:
        f (function) - Objective function 
        x0 (vector) - Starting coordinate 
        gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details
        lineSearchMethod (LineSearchMethod)- LineSearchMethod struct impementing lineSearchInterface with a specific step length algorithm 
        alpha (Float64) - Initial step length
        tol (Float64) - Stop criteria, if norm grad smaller than tolerance value iterations will stop
        lim (Int) - Maximum number of iterations

    Output - named tuple with the following fields
        minimum (Vector)- Final coordinate 
        path (Array) - Coordinates at each iteration 
        gradients (Array) - Gradient values at each iteration
        functionValues (Array) - Objective function values at each iteration
        iterations (Int) - Bumber of iterations 
    """
    x = copy(x0)
    fCur = f(x0)

    #TODO put step search data into algorithmData? right now only track iter step
    algorithmData = track ? 
        AlgorithmData(lim; lineSearchIterCount = Int) :
        NoAlgorithmData()

    logger = initLogger(track, x0, fCur, lim, algorithmData = algorithmData)

    xOld = copy(x)
    fOld = fCur

    for i in 1:lim
        #TODO check if gradient is used in convergence, if not then don't calculate it
        gradRes = gradient(gradEstimator, f, x)
        grad = gradRes.grad
        # funcEval.

        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, 0)
        if converged 
            setConvergenceReason!(logger, reason)
            finalizeLogger!(logger)
        
            if printIter
                println("Line search converged in $i iterations.")
            end
            return (
                minimumPoint = x,
                finalValue = fCur,
                logger = logger
            )
        end

        xOld = copy(x)
        fOld = fCur
        
        stepSearchResult = stepSearch(lineSearchMethod, gradEstimator, f, -grad, x, fCur, grad, alpha, lineSearchLim)
        
        x = stepSearchResult.xNew
        fCur = stepSearchResult.fFinal
        #TODO right now grad evals + 1 for convergence check, fix if make gradEval dependent on convergenceCrit
        logIter!(logger, fCur, x, grad, stepSearchResult.alphaFinal, stepSearchResult.funcEvals, stepSearchResult.gradEvals + 1 + gradRes.funcEvals, gradRes.funcEvals, lineSearchIterCount=stepSearchResult.lineSearchIterCount)
        

    end
    error("line search failed: maximum iterations ($lim) reached")
end

