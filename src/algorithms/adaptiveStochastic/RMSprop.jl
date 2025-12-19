```
Resource:
Dive into Deep learning - https://d2l.ai/chapter_optimization/rmsprop.html
```

struct RMSprop <: AdaptiveMethod
    eta::Float64   # learning rate
    gamma::Float64   # decay rate
    epsilon::Float64   # numerical stability
end

RMSprop(eta; gamma=0.9, epsilon=1e-8) = RMSprop(eta, gamma, epsilon)

function initState(parameters::RMSprop, x0)
    return (
        v = zeros(length(x0))  # squared gradient accumulator
    )
end

function stepUpdate(parameters::RMSprop, gradEstimator, f, xCur, fCur, gradCur, state, iteration)
    v = parameters.gamma .* state.v .+ (1 - parameters.gamma) .* gradCur.^2
    
    step = parameters.eta .* gradCur ./ (sqrt.(v) .+ parameters.epsilon)
    
    return (step, (v=v,))
end