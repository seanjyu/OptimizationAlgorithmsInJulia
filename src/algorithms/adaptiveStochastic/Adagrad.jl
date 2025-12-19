```
Resource
Dive into Deep Learning - https://d2l.ai/chapter_optimization/adagrad.html
```


struct Adagrad <: AdaptiveMethod
    eta::Float64   # learning rate
    epsilon::Float64   # numerical stability
end

Adagrad(eta; epsilon=1e-8) = Adagrad(eta, epsilon)

function initState(parameters::Adagrad, x0)
    return (
        s = zeros(length(x0))  # sum of squared gradients
    )
end

function stepUpdate(parameters::Adagrad, gradEstimator, f, xCur, fCur, gradCur, state, iteration)
    s = state.s .+ gradCur.^2
    
    step = parameters.eta .* gradCur ./ (sqrt.(s) .+ parameters.epsilon)
    
    return (step, (s=s))
end