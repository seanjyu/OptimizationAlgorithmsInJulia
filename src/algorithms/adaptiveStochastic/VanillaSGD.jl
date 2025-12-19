struct VanillaSGD <: AdaptiveMethod
    η::Float64  # learning rate
end

function initState(opt::VanillaSGD, x0)
    return nothing
end

function stepUpdate(opt::VanillaSGD, gradEstimator, f, xCur, fCur, gradCur, state, iteration)
    step = opt.η .* gradCur
    return (step, nothing)
end