struct UOWNewtonOpt <: UnconstrainedOptMethod
    gradEstimator::GradientEstimator
    c::ConvergenceCriteria
    
end

function solveUnconstrainedOpt(f, x0, p::UOWNewtonOpt)
    return NewtonMethod(f, x0, p.gradEstimator, p.c)
end
