 """
More-Thuente Line Search 

Reference(s)

Hyperparameters
    c1 (Float64) - Amrijo condition coefficient 
    c2 (Float64) - Curvature coefficient
    stepMin
    stepMax
    xtrapf
    widthTol
    bisectionThreshold
    bracketBoundFactor
    stepLengthStrategy (StepLengthStrategy) - specific step length strategy (e.g. Backtracking, QuadraticInterpolation, etc)
    strong (Bool) - Strong Wolfe condition flag, if true strong Wolfe conditions will be used 
    printLineSearchCount (Bool) - flag to print number of line search iterations until convergence 
    curvTol (Float64) - Tolerance for curvature condition
    directionTol (Float64) - Tolerance for descent direction condition
"""
struct MoreThuenteLineSearch <: LineSearchMethod
    c1::Float64
    c2::Float64
    stepMin::Float64
    stepMax::Float64
    xtrapf::Float64
    widthTol::Float64
    bisectionThreshold::Float64
    bracketBoundFactor::Float64
    printLineSearchCount::Bool
    curvTol::Float64
    directionTol::Float64

    # default constructor
    MoreThuenteLineSearch(;c1 = 1e-4, c2 = 0.1, stepMin = 1e-10, stepMax = 1e5, xtrapf = 4, widthTol = 1e-5, bisectionThreshold = 0.66, bracketBoundFactor = 0.66, printLineSearchCount = false, curvTol = 1e-8, directionTol = 1e-6) = new(c1, c2, stepMin, stepMax, xtrapf, widthTol, bisectionThreshold, bracketBoundFactor, printLineSearchCount, curvTol, directionTol)
end 

"""
stepSearch - Concrete implementation of More Thuente line search
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
function stepSearch(parameters::MoreThuenteLineSearch, 
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

    # initialize variables
    alphaCur = alpha
    funcEvals = 0
    gradEvals = 0

    # Bracket Values
    alphaBest = 0
    alphaOther = 0
    fBest = fCur                     # φ(alphaBest)
    fOther = fCur                     # φ(alphaOther)
    gradBest = dirDerivative            # φ'(alphaBest)
    gradOther = dirDerivative            # φ'(alphaOther)

    # state variables
    stage1 = true
    bracketed = false
    lineSearchIterCount = 0

    # For bisection safeguard - currently using factor of 2 
    prevWidth = 2 * (parameters.stepMax - parameters.stepMin)

    # preallocate xPropose for memory
    if xCur isa AbstractArray
        xPropose = similar(xCur)
    end

    while lineSearchIterCount < lineSearchLim

        # set min and max steps (left and right side of bracket)
        if bracketed
            alphaMin = max(min(alphaBest, alphaOther), parameters.stepMin)
            alphaMax = min(max(alphaBest, alphaOther), parameters.stepMax)
        else
            alphaMin = max(alphaBest, parameters.stepMin)
            alphaMax = min(alphaCur + parameters.xtrapf * (alphaCur - alphaBest), parameters.stepMax)
        end

        # ensure current alpha within bound
        alphaCur = clamp(alphaCur, 
                        max(alphaMin, parameters.stepMin), 
                        min(alphaMax, parameters.stepMax))

        # Unusual termination check
        # if somehow current step is smaller than minimum or larger than maximum or that the width of bracket 
        # is too small terminate early
        if  (bracketed && (alphaCur <= alphaMin || alphaCur >= alphaMax)) || 
            (bracketed && (alphaMax - alphaMin <= parameters.widthTol * alphaMax))
            # @. xPropose = xCur + alphaBest * direction

            if xCur isa AbstractArray
                @. xPropose = xCur + alphaBest * direction
            else
                xPropose = xCur + alphaBest * direction
            end
            
            return (xNew = copy(xPropose), fFinal = fBest, alphaFinal = alphaBest, 
                    funcEvals = funcEvals, gradEvals = gradEvals, 
                    lineSearchIterCount = lineSearchIterCount)    
        end

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

        # Armijo Condition Values
        armijoDerivativeConditionValue = parameters.c1 * dirDerivative
        armijoConditionValue = fCur + alphaCur * armijoDerivativeConditionValue

        armijoCondition = fPropose <= armijoConditionValue
        curvatureCondition = abs(dirDerivativePropose) <= parameters.c2 * abs(dirDerivative) + parameters.curvTol 
                
        if armijoCondition && curvatureCondition
            if parameters.printLineSearchCount
                println("Line search converged in $lineSearchIterCount iterations")
            end
            return (xNew = copy(xPropose), fFinal = fPropose, alphaFinal = alphaCur, funcEvals = funcEvals, gradEvals = gradEvals, lineSearchIterCount = lineSearchIterCount)
        else
            # check stage
            # If armijo condition pass and relaxed curvature condition (basically checking past steepest descent)
            # then move onto second stage ()
            if stage1 && fPropose <= armijoConditionValue && dirDerivativePropose >= min(parameters.c1, parameters.c2) * dirDerivative
                stage1 = false
            end
            
            # if still in stage 1 and fPropose is lower than fCur but armijo condition is not met yet
            if stage1 && fPropose <= fBest && fPropose > armijoConditionValue
                # compute More Thuente Auxiliary function
                fTrialAux = fPropose - alphaCur * armijoDerivativeConditionValue
                fBestAux = fBest - alphaBest * armijoDerivativeConditionValue
                fOtherAux = fOther - alphaOther * armijoDerivativeConditionValue
                gradTrialAux = dirDerivativePropose - armijoDerivativeConditionValue
                gradBestAux = gradBest - armijoDerivativeConditionValue
                gradOtherAux = gradOther - armijoDerivativeConditionValue
                
                # Call cstep with ψ values
                (alphaBest, alphaOther, fBestAux, fOtherAux, gradBestAux, gradOtherAux, alphaCur, bracketed) = updateMoreThuenteBracket(
                    parameters,
                    alphaBest, fBestAux, gradBestAux,
                    alphaOther, fOtherAux, gradOtherAux,
                    alphaCur, fTrialAux, gradTrialAux,
                    bracketed, alphaMin, alphaMax
                )

                # Transform back to φ values
                fBest = fBestAux + alphaBest * armijoDerivativeConditionValue
                fOther = fOtherAux + alphaOther * armijoDerivativeConditionValue
                gradBest = gradBestAux + armijoDerivativeConditionValue
                gradOther = gradOtherAux + armijoDerivativeConditionValue

            else
                # stage 2 - condtions are met, use phi directly
                (alphaBest, alphaOther, fBest, fOther, gradBest, gradOther, alphaCur, bracketed) = updateMoreThuenteBracket(
                    parameters,
                    alphaBest, fBest, gradBest,
                    alphaOther, fOther, gradOther,
                    alphaCur, fPropose, dirDerivativePropose,
                    bracketed, alphaMin, alphaMax
                )
            end
            
            # bisection safeguard 
            if bracketed
                currentWidth = abs(alphaOther - alphaBest)
                if currentWidth >= parameters.bisectionThreshold * prevWidth
                    # Force bisection
                    alphaCur = alphaBest + 0.5 * (alphaOther - alphaBest)
                end
                prevWidth = currentWidth
            end

            lineSearchIterCount += 1
            
        end
    end

    # If count reaches limit throw error
    error("MoreThuente linesearch failed: maximum iterations ($lineSearchLim) reached")
end

function updateMoreThuenteBracket(parameters::MoreThuenteLineSearch, alphaBest, fBest, gradBest, alphaOther, fOther, gradOther, alphaCur, fTrial, gradTrial, bracketed, alphaMin, alphaMax)
    # check sign for derivative
    derivativeSign = gradTrial * (gradBest / abs(gradBest)) 

    # first case - trial point has higher function than current best point
    if fTrial > fBest
        bound = true
        bracketed = true
        theta = 3 * (fBest - fTrial) / (alphaCur - alphaBest) + gradBest + gradTrial
        s = norm([theta, gradBest, gradTrial], Inf)
        gamma = s * sqrt((theta / s)^2 - (gradBest / s) * (gradTrial / s))
        if (alphaCur < alphaBest)
            gamma = -gamma
        end
        p = (gamma - gradBest) + theta
        q = ((gamma - gradBest) + gamma) + gradTrial
        r = p/q

        alphaCubic = alphaBest + r * (alphaCur - alphaBest)
        alphaQuadratic = alphaBest + ((gradBest / ((fBest - fTrial) / (alphaCur - alphaBest) + gradBest)) / 2) * (alphaCur - alphaBest)

        if abs(alphaCubic - alphaBest) < abs(alphaQuadratic - alphaBest)
            alphaFinal = alphaCubic
        else
            alphaFinal = alphaCubic + (alphaQuadratic - alphaCubic) / 2
        end

    # second case
    elseif derivativeSign < 0
        bound = false
        bracketed = true

        theta = 3 * (fBest - fTrial) / (alphaCur - alphaBest) + gradBest + gradTrial
        s = norm([theta, gradBest, gradTrial], Inf)
        gamma = s * sqrt((theta / s)^2 - (gradBest / s) * (gradTrial / s))
        if alphaCur > alphaBest
            gamma = -gamma
        end
        p = (gamma - gradTrial) + theta
        q = ((gamma - gradTrial) + gamma) + gradBest
        r = p/q
        
        alphaCubic = alphaCur + r * (alphaBest - alphaCur)
        alphaSecant = alphaCur + (gradTrial / (gradTrial - gradBest)) * (alphaBest - alphaCur)

        if abs(alphaCubic - alphaCur) >= abs(alphaSecant - alphaCur)
            alphaFinal = alphaCubic
        else
            alphaFinal = alphaSecant
        end

    # case 3
    elseif abs(gradTrial) < abs(gradBest)
        bound = true

        theta = 3 * (fBest - fTrial) / (alphaCur - alphaBest) + gradBest + gradTrial
        s = norm([theta, gradBest, gradTrial], Inf)
        gamma = s * sqrt(max(0, (theta / s)^2 - (gradBest / s) * (gradTrial / s)))

        if alphaCur > alphaBest
            gamma = -gamma
        end
        
        p = (gamma - gradTrial) + theta
        q = (gamma + (gradBest - gradTrial)) + gamma
        r = p / q

        if r < 0 && gamma != 0
            alphaCubic = alphaCur + r * (alphaBest - alphaCur)
        elseif alphaCur > alphaBest
            alphaCubic = alphaMax
        else
            alphaCubic = alphaMin
        end

        alphaSecant = alphaCur + (gradTrial / (gradTrial - gradBest)) * (alphaBest - alphaCur)

        if bracketed
            # Already bracketed: prefer closer to alphaCur
            if abs(alphaCur - alphaCubic) < abs(alphaCur - alphaSecant)
                alphaFinal = alphaCubic
            else
                alphaFinal = alphaSecant
            end
        else
            # Not bracketed: prefer farther from alphaCur (explore more)
            if abs(alphaCur - alphaCubic) > abs(alphaCur - alphaSecant)
                alphaFinal = alphaCubic
            else
                alphaFinal = alphaSecant
            end
        end
    # case 4
    else 
        bound = false
        if bracketed
            theta = 3 * (fTrial - fOther) / (alphaOther - alphaCur) + gradOther + gradTrial
            s = norm([theta, gradOther, gradTrial], Inf)
            gamma = s * sqrt((theta / s)^2 - (gradOther / s) * (gradTrial / s))
            if alphaCur > alphaOther
                gamma = -gamma
            end
            p = (gamma - gradTrial) + theta
            q = ((gamma - gradTrial) + gamma) + gradOther
            r = p / q
            alphaCubic = alphaCur + r * (alphaOther - alphaCur)
            alphaFinal = alphaCubic
        elseif alphaCur > alphaBest
            alphaFinal = alphaMax
        else
            alphaFinal = alphaMin
        end
    end

    if fTrial > fBest
        # Case 1: trial is worse, it becomes the other bracket endpoint
        alphaOther = alphaCur
        fOther = fTrial
        gradOther = gradTrial
    else
        # Cases 2-4: trial is better
        if derivativeSign < 0  # gradTrial * gBest < 0, opposite signs
            # Old best becomes other bracket endpoint
            alphaOther = alphaBest
            fOther = fBest
            gradOther = gradBest
        end
        # Trial becomes new best
        alphaBest = alphaCur
        fBest = fTrial
        gradBest = gradTrial
    end

    if bracketed && bound
        if alphaOther > alphaBest
            alphaFinal = min(alphaBest + parameters.bracketBoundFactor * (alphaOther - alphaBest), alphaFinal)
        else
            alphaFinal = max(alphaBest + parameters.bracketBoundFactor * (alphaOther - alphaBest), alphaFinal)
        end
    end

    return (alphaBest, alphaOther, fBest, fOther, gradBest, gradOther, alphaFinal, bracketed)
end