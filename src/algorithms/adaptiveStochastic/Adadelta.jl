```
Resource 
Dive into deep learning - https://d2l.ai/chapter_optimization/adadelta.html
```

struct Adadelta <: AdaptiveMethod
    rho::Float64   
    epsilon::Float64   # numerical stability
end

Adadelta(eta; epsilon=1e-8) = Adadelta(eta, epsilon)

function initState(parameters::Adadelta, x0)
    return (
        s = zeros(length(x0)),  #  squared gradients
        prevDeltaX = zeros(length(x0))
    )
end

function stepUpdate(parameters::Adadelta, gradEstimator, f, xCur, fCur, gradCur, state, iteration)
    s = parameters.rho * state.s .+ (1 - parameters.rho) .* gradCur.^2

    scaledG = sqrt.(state.prevDeltaX + parameters.epsilon) ./ sqrt.(s + parameters.epsilon) .* gradCur 

    deltaX = parameters.rho * state.prevDeltaX + (1 - parameters.rho) * scaledG.^2
    
    step = scaledG
    
    return (step, (s=s, prevDeltaX = deltaX))
end