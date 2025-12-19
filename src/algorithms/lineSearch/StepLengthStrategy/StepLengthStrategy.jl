"""

"""

abstract type StepLengthStrategy end

"""
calculateStepLength - Update step length based on strategy
Returns new alpha value, 
"""
function calculateStepLength(strategy::StepLengthStrategy, 
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
    error("calculateStepLength not implemented for $(typeof(strategy))")                            
end

