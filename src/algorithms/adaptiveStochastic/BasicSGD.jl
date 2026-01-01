struct BasicSGD <: AdaptiveMethod
    eta::Float64  # learning rate
end

function initState(opt::BasicSGD, x0)
    return nothing
end

function stepUpdate(opt::BasicSGD, gradEstimator, f, xCur, fCur, gradCur, state, iteration)
    step = opt.eta .* gradCur
    return (step, nothing)
end