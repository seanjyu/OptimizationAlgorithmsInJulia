module armijoLineSearchModule
    using ..lineSearchInterface: LineSearchMethod
    import ..lineSearchInterface: stepSearch
    using ..GradientEstimatorInterface: GradientEstimator
    using LinearAlgebra 
    export ArmijoLineSearchBt

    """
    Amrijo Line Search

    Reference(s)
        The stepSearch implementation generally follows algorithm 3.1 (pg 37) from 'Numerical Optimization' by Nocedal and Wright.
        The Armijo condition can be found in equation 3.4 in the same text (pg 33). 
        Note the conditions have been adapted slightly to allow for some tolerance as small numbers may cause the inequalities to fail.
        Also, the gradient estimator is not used in this function but is still in the input due to the method signature (for easier generalization).

    Hyperparameters
        c1 (Float64) - Amrijo condition coefficient 
        rho (Float64) - Backtracking coefficient 
    """
    struct ArmijoLineSearchBt <: LineSearchMethod
        c1::Float64
        rho::Float64

        # default constructor
        ArmijoLineSearchBt(;c1 = 1e-4, rho = 0.5) = new(c1, rho)
    end
    
    """
    stepSearch - concrete implementation of Armijo Condition Backtracking line search
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

    """
    function stepSearch(parameters::ArmijoLineSearchBt, 
                        gradEstimator::GradientEstimator, 
                        f, 
                        direction, 
                        xCur, 
                        fCur,
                        gradCur, 
                        alpha,  
                        lineSearchLim)
        
        dirDerivative = dot(direction, gradCur)
        alphaCur = alpha
        btCount = 0
        while btCount < lineSearchLim
            xPropose = xCur .+ alphaCur .* direction
            fPropose = f(xPropose)

            armijoCondition = fPropose <= fCur + parameters.c1 * alphaCur * dirDerivative

            if armijoCondition
                return (xNew = xPropose, fFinal = fPropose, alphaFinal = alphaCur)
            else
                alphaCur *= parameters.rho
                btCount += 1
            end
        end

        # If bt count reaches limit throw error
        error("Armijo backtracking failed: maximum iterations ($lineSearchLim) reached")
    end
end
