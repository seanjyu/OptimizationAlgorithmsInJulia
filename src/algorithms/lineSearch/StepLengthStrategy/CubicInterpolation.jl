struct CubicInterpolation <: StepLengthStrategy
    rhoMin::Float64
    rhoMax::Float64
    tol::Float64

    CubicInterpolation(;rhoMin = 1e-3, rhoMax = 0.5, tol = 1e-10) = new(rhoMin, rhoMax, tol)
end

function calculateStepLength(parameters::CubicInterpolation,
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


    #TODO implement calculate step length
    alpha = 0
    funcEvals = 0
    gradEvals = 0
    return (alpha = alpha, funcEvals = 0, gradEvals = 0) 
end