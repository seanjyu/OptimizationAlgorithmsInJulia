"""
Armijo Line Search

Reference(s)
    The stepSearch implementation generally follows algorithm 3.1 (pg 37) from 'Numerical Optimization' by Nocedal and Wright.
    The Armijo condition can be found in equation 3.4 in the same text (pg 33). 
    Note the conditions have been adapted slightly to allow for some tolerance as small numbers may cause the inequalities to fail.
    Also, the gradient estimator is not used in this function but is still in the input due to the method signature (for easier generalization).

Hyperparameters
    c1 (Float64) - Amrijo condition coefficient
    stepLengthStrategy (StepLengthStrategy) - specific step length strategy (e.g. Backtracking, QuadraticInterpolation, etc)
    printLineSearchCount (Bool) - flag to print number of line search iterations until convergence 
    directionTol (Float64) - Tolerance for descent direction condition
"""
struct ArmijoLineSearch <: LineSearchMethod
    c1::Float64
    stepLengthStrategy::StepLengthStrategy
    printLineSearchCount::Bool
    directionTol::Float64

    # default constructor
    ArmijoLineSearch(;c1 = 1e-4, stepLengthStrategy = Backtracking(rho=0.5), printLineSearchCount = false, directionTol = 1e-6) = new(c1, stepLengthStrategy, printLineSearchCount, directionTol)
end

"""
stepSearch 
    Concrete implementation of Armijo Condition Backtracking line search
    Follows method interface defined in lineSearchBase.jl
Inputs
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
    funcEvals (int) - number of function calls
"""
function stepSearch(parameters::ArmijoLineSearch, 
                    gradEstimator::GradientEstimator, 
                    f, 
                    direction, 
                    xCur, 
                    fCur,
                    gradCur, 
                    alpha,  
                    lineSearchLim)
    
    dirDerivative = dot(direction, gradCur)

    # check direction is a valid descent direction
    if dirDerivative >= parameters.directionTol
        error("Not a descent direction: directional derivative = $dirDerivative")
    end

    alphaCur = alpha
    lineSearchIterCount = 0
    funcEvals = 0
    gradEvals = 0
    alphaPrev = nothing
    fProposePrev = nothing
    
    # preallocate xPropose for memory
    if xCur isa AbstractArray
        xPropose = similar(xCur)
    end

    while lineSearchIterCount < lineSearchLim
        if xCur isa AbstractArray
        # Vector case
            @. xPropose = xCur + alphaCur * direction
        else
        # Scalar case
            xPropose = xCur + alphaCur * direction
        end
        fPropose = f(xPropose)
        # gradPropose = gradProposeRes.grad
        # gradEvals += 1
        # dirDerivativePropose = dot(gradPropose, direction)
        funcEvals += 1

        armijoCondition = fPropose <= fCur + parameters.c1 * alphaCur * dirDerivative

        #TODO need to rethink the dirDerivativePropose in the calculateStepLength
        if armijoCondition
            return (xNew = xPropose, fFinal = fPropose, alphaFinal = alphaCur, funcEvals = funcEvals, gradEvals = 0, lineSearchIterCount = lineSearchIterCount)
        else
            alphaSearchResult = calculateStepLength(parameters.stepLengthStrategy,
                                                    f,
                                                    gradEstimator,
                                                    direction,
                                                    xCur,
                                                    fCur,
                                                    fPropose,
                                                    fProposePrev,
                                                    alphaCur,
                                                    alphaPrev,
                                                    dirDerivative,
                                                    0 #dirDerivativePropose
                                                    )
            lineSearchIterCount += 1
            alphaCur = alphaSearchResult.alpha
            funcEvals += alphaSearchResult.funcEvals
            gradEvals += alphaSearchResult.gradEvals

            # Update history for next iteration
            alphaPrev = alphaCur
            fProposePrev = fPropose
            alphaCur = alphaSearchResult.alpha
        end
    end

    # If count reaches limit throw error
    error("Armijo linesearch failed: maximum iterations ($lineSearchLim) reached")
end