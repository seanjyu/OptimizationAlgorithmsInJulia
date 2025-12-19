```
Resource
Dive into Deep Learning - https://d2l.ai/chapter_optimization/adam.html
```

struct Adam <: AdaptiveMethod
    eta::Float64   # learning rate
    beta1::Float64  # first moment decay
    beta2::Float64  # second moment decay
    epsilon::Float64   # numerical stability
end

Adam(eta; beta1=0.9, beta2=0.999, epsilon=1e-8) = Adam(eta, beta1, beta2, epsilon)

function initState(parameters::Adam, x0)
    return (
        v = zeros(length(x0)),  # first moment
        s = zeros(length(x0))   # second moment
    )
end

function stepUpdate(parameters::Adam, gradEstimator, f, xCur, fCur, gradCur, state, iteration)
    v, s = state.v, state.s
    
    # Update moments
    v .= parameters.beta1 .* m .+ (1 - parameters.beta1) .* gradCur
    s .= parameters.beta2 .* v .+ (1 - parameters.beta2) .* gradCur.^2
    
    # Bias correction
    vHat = v ./ (1 - parameters.beta1^iteration)
    sHat = s ./ (1 - parameters.beta2^iteration)
    
    # Compute step
    step = parameters.eta .* vHat ./ (sqrt.(sHat) .+ parameters.epsilon)
    
    return (step, (m=m, v=v))
end