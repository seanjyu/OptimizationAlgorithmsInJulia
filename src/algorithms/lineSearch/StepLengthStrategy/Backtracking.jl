struct Backtracking <: StepLengthStrategy
    rho::Float64
    Backtracking(;rho=0.5) = new(rho)
    # Positional constructor (add this)
    Backtracking(rho::Float64) = new(rho)
end

function calculateStepLength(parameters::Backtracking,
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
    alpha = parameters.rho * alphaCur
    return (alpha = alpha, funcEvals = 0, gradEvals = 0)
end