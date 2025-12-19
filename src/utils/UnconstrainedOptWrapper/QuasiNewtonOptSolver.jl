struct UOWQuasiNewtonOpt <: UnconstrainedOptMethod
    gradEstimator::GradientEstimator
    quasiNewtonMethod::QuasiNewtonMethod
    lineSearchMethod::LineSearchMethod
    c::ConvergenceCriteria
    # TODO add default constructor, note that the original Quasi Newton method already has default values
end

function solveUnconstrainedOpt(f, x0, parameters::UOWQuasiNewtonOpt)
    return QuasiNewtonOpt(f, x0, parameters.gradEstimator, parameters.quasiNewtonMethod, parameters.lineSearchMethod, parameters.c)
end