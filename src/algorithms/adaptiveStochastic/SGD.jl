abstract type AdaptiveMethod end

abstract type state end

function initState(adaptiveMethod::AdaptiveMethod, x0)
end

function stepUpdate(adaptiveMethod::AdaptiveMethod, gradEstimator::GradientEstimator, f, xCur, fCur, gradCur, state, iter)
end


function SGD(f, x0, gradEstimator::GradientEstimator, rule::AdaptiveMethod;
             epochs = 100,
             stepsPerEpoch = 100,
             track = true)
    
    x = copy(x0)
    state = initState(rule, x0)
    totalSteps = epochs * stepsPerEpoch

    algorithmData = track ? 
        AlgorithmData(lim; states=typeof(state)) : 
        NoAlgorithmData()
    
    logger = initLogger(track, x0, f(x0), totalSteps, algorithmData = algorithmData)
    
    for epoch in 1:epochs
        for t in 1:stepsPerEpoch
            iteration = (epoch - 1) * stepsPerEpoch + t
            
            
            gradRes = gradient(gradEstimator, f, x)
            step, state = computeStep(rule, gradRes.grad, state, iteration)
            
            x = x .- step

            #TODO add logIter
        end
    end
    
    fFinal = f(x)
    finalizeLogger!(logger)
    
    return (
        minimum = x,
        finalValue = fFinal,
        logger = logger
    )
end