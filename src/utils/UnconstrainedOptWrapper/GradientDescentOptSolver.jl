struct UOWGradientDescent <: UnconstrainedOptMethod
    gradEstimator::GradientEstimator
    c::ConvergenceCriteria
    track::Bool
end

function solveUnconstrainedOpt(f, x0, p::UOWGradientDescentOpt)
    return GradientDescent(f, x0, p.gradEstimator, p.c)
end
