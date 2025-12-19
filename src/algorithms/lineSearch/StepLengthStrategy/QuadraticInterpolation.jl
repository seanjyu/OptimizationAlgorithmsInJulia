struct QuadraticInterpolation <: StepLengthStrategy
    rhoMin::Float64
    rhoMax::Float64
    tol::Float64

    QuadraticInterpolation(;rhoMin = 1e-3, rhoMax = 0.5, tol = 1e-10) = new(rhoMin, rhoMax, tol)
end

function calculateStepLength(parameters::QuadraticInterpolation,
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
                             dirDerivativePropose, 
                             ) 

                             
    a = (fPropose - fCur - dirDerivative * alphaCur) / (alphaCur^2)
    if abs(a) < parameters.tol
        alpha = parameters.rhoMax * alphaCur
    else
        alpha = -dirDerivative / (2 * a)
        alpha = clamp(alpha, parameters.rhoMin * alphaCur, parameters.rhoMax * alphaCur)
    end
    
    return (alpha = alpha, funcEvals = 0, gradEvals = 0)
end