module WolfeLineSearchModule
    using ..lineSearchInterface: LineSearchMethod
    import ..lineSearchInterface: stepSearch
    using ..GradientEstimatorInterface: GradientEstimator, gradient
    using LinearAlgebra 
    export WolfeBTLineSearch

    """
    Wolfe Condition Backtracking Line Search 

    Reference(s)
        This implementation generally follows algorithm 3.1 (pg 37) from 'Numerical Optimization' by Nocedal and Wright.
        The Wolfe conditions can be found in equations 3.6 and 3.7 (strong form) in the same text (pg 34). 
        Note the conditions have been adapted slightly to allow for some tolerance as small numbers may cause the inequalities to fail.

    Hyperparameters
        c1 (Float64) - Amrijo condition coefficient 
        c2 (Float64) - Curvature coefficient
        rho (Float64) - Backtracking coefficient
        strong (Bool) - Strong Wolfe condition flag, if true strong Wolfe conditions will be used 
        printBTIter (Bool) - flag to print number of backtracking iterations until convergence 
        curvTol (Float64) - Tolerance for curvature condition
        directionTol (Float64) - Tolerance for descent direction condition
    """
    struct WolfeBTLineSearch <: LineSearchMethod
        c1::Float64
        c2::Float64
        rho::Float64
        strong::Bool
        printBTIter::Bool
        curvTol::Float64
        directionTol::Float64

        # default constructor
        WolfeBTLineSearch(;c1 = 1e-4, c2 = 0.1, rho = 0.5, strong = false, printBTIter = false, curvTol = 1e-8, directionTol = 1e-6) = new(c1, c2, rho, strong, printBTIter, curvTol, directionTol)
    end 

    """
    stepSearch - Concrete implementation of Wolfe Condition Backtracking line search
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
    function stepSearch(parameters::WolfeBTLineSearch, 
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

        # backtracking loop
        alphaCur = alpha
        btCount = 0
        while btCount < lineSearchLim
            xPropose = xCur .+ alphaCur .* direction
            fPropose = f(xPropose)

            gradPropose = gradient(gradEstimator, f, xPropose)
            dirDerivativePropose = dot(gradPropose, direction)

            armijoCondition = fPropose <= fCur + parameters.c1 * alphaCur * dirDerivative
            curvatureCondition = parameters.strong ? 
                    abs(dirDerivativePropose) <= parameters.c2 * abs(dirDerivative) + parameters.curvTol :
                    dirDerivativePropose >= parameters.c2 * dirDerivative - parameters.curvTol
                    
            if armijoCondition && curvatureCondition
                if parameters.printBTIter
                    println("Backtracking converged in $btCount iterations")
                end
                return (xNew = xPropose, fFinal = fPropose, alphaFinal = alphaCur)
            else
                # println("$dirDerivativePropose, $(parameters.c2 * dirDerivative - parameters.curvTol), $alphaCur")
                alphaCur *= parameters.rho
                btCount += 1
            end
        end

        # If bt count reaches limit throw error
        error("Wolfe Condition backtracking failed: maximum iterations ($lineSearchLim) reached")
    end
end