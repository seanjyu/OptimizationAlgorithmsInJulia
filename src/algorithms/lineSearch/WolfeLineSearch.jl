 """
Wolfe Condition Line Search 

Reference(s)
    This implementation generally follows algorithm 3.1 (pg 37) from 'Numerical Optimization' by Nocedal and Wright.
    The Wolfe conditions can be found in equations 3.6 and 3.7 (strong form) in the same text (pg 34). 
    Note the conditions have been adapted slightly to allow for some tolerance as small numbers may cause the inequalities to fail.

Hyperparameters
    c1 (Float64) - Amrijo condition coefficient 
    c2 (Float64) - Curvature coefficient
    stepLengthStrategy (StepLengthStrategy) - specific step length strategy (e.g. Backtracking, QuadraticInterpolation, etc)
    strong (Bool) - Strong Wolfe condition flag, if true strong Wolfe conditions will be used 
    printLineSearchCount (Bool) - flag to print number of line search iterations until convergence 
    curvTol (Float64) - Tolerance for curvature condition
    directionTol (Float64) - Tolerance for descent direction condition
"""
    struct WolfeLineSearch <: LineSearchMethod
        c1::Float64
        c2::Float64
        stepLengthStrategy::StepLengthStrategy
        strong::Bool
        printLineSearchCount::Bool
        curvTol::Float64
        directionTol::Float64

        # default constructor
        WolfeLineSearch(;c1 = 1e-4, c2 = 0.1, stepLengthStrategy = Backtracking(rho=0.5), strong = false, printLineSearchCount = false, curvTol = 1e-8, directionTol = 1e-6) = new(c1, c2, stepLengthStrategy, strong, printLineSearchCount, curvTol, directionTol)
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
    function stepSearch(parameters::WolfeLineSearch, 
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
        funcEvals = 0
        gradEvals = 0
        alphaPrev = nothing
        fProposePrev = nothing
        lineSearchIterCount = 0

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
            funcEvals += 1

            gradProposeRes = gradient(gradEstimator, f, xPropose)
            gradPropose = gradProposeRes.grad
            gradEvals += 1
            dirDerivativePropose = dot(gradPropose, direction)

            armijoCondition = fPropose <= fCur + parameters.c1 * alphaCur * dirDerivative
            curvatureCondition = parameters.strong ? 
                    abs(dirDerivativePropose) <= parameters.c2 * abs(dirDerivative) + parameters.curvTol :
                    dirDerivativePropose >= parameters.c2 * dirDerivative - parameters.curvTol
                    
            if armijoCondition && curvatureCondition
                if parameters.printLineSearchCount
                    println("Line search converged in $lineSearchIterCount iterations")
                end
                return (xNew = copy(xPropose), fFinal = fPropose, alphaFinal = alphaCur, funcEvals = funcEvals, gradEvals = gradEvals, lineSearchIterCount = lineSearchIterCount)
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
                                                        dirDerivativePropose)

                
                # Update history for next iteration
                alphaPrev = alphaCur
                fProposePrev = fPropose

                alphaCur = alphaSearchResult.alpha
                funcEvals += alphaSearchResult.funcEvals
                gradEvals += alphaSearchResult.gradEvals
                
                lineSearchIterCount += 1
                if parameters.printLineSearchCount
                    println("\n--- Iteration $lineSearchIterCount ---")
                    println("alphaCur: $alphaCur, alphaPrev: $alphaPrev")
                    println("fCur: $fCur, fPropose: $fPropose")
                    println("Armijo: $armijoCondition, Curvature: $curvatureCondition")
                end
                
            end
        end

        # If count reaches limit throw error
        error("Wolfe linesearch failed: maximum iterations ($lineSearchLim) reached")
    end
