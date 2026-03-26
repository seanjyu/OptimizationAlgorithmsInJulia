"""
Interface type for Adaptive Stochastic methods
    Type: 
        AdaptiveStochasticMethod
        
    Function(s):
        initState
        stepUpdate - Method interface to perform adaptive step update based on specific adaptive stochastic method
"""
abstract type AdaptiveStochasticMethod end

abstract type state end

function initState(AdaptiveStochasticMethod::AdaptiveStochasticMethod, x0)
end

function stepUpdate(AdaptiveStochasticMethod::AdaptiveStochasticMethod, gradEstimator::GradientEstimator, f, xCur, fCur, gradCur, state, iter)
end



"""
SGD

Required Inputs     
    f (function) - Objective function 
    x0 (vector) - Starting coordinate 
    gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details    
    criteria (Criteria) - 

Optional Inputs
    lim (Int) - Maximum number of iterations
    track (boolean) - Flag whether or not to track variables from iteration 

Output - named tuple with the following fields
    minimumPoint (Vector) - Final coordinate of algorithm 
    finalValue (Float64) - Objective function evaluation at final coordinate 
    logger (struct) - if track flag set to true then the following fields can be accessed
        path (Array) - Coordinates at each iteration 
        gradients (Array) - Gradient values at each iteration
        functionValues (Array) - Objective function values at each iteration
        iterations (Int) - Number of iterations 
        algorithmData (Struct) - Algorithm specific data
            SOMEFIELD - FIELDDESCR

"""

function SGD(f, x0, gradEstimator::GradientEstimator, rule::AdaptiveStochasticMethod;
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
        minimumPoint = x,
        finalValue = fFinal,
        logger = logger
    )
end
