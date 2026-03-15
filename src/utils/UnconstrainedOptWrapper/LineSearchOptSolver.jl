struct UOWLineSearch <: UnconstrainedOptMethod
    # check line search file for parameters
    gradEstimator::GradientEstimator
    c::ConvergenceCriteria
    track::Bool
end


function solveUnconstrainedOpt(f, x0, p::UOWLineSearch)
    # call/return line search 
end