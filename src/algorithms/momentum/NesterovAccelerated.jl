function NesterovAcceleratedGradient(f, x0, gradEstimator::GradientEstimator, criteria::ConvergenceCriteria; alpha=0.1, beta = 0.8, tol=1e-5, lim=100, track = true)

    x = copy(x0)
    fCur = f(x0)

    prevX = copy(x0)

    momentumType = x0 isa Number ? Float64 : Vector{Float64}
    
    algorithmData = track ? 
        AlgorithmData(lim; momentums=momentumType) : 
        NoAlgorithmData()

    logger = initLogger(track, x0, fCur, lim, algorithmData = algorithmData)

    for i in 1:lim

        xOld = copy(x)
        fOld = fCur
        
        momentum = beta .* (x .- prevX)
        lookAheadRes = gradient(gradEstimator, f, x + momentum)
        lookAheadGrad = lookAheadRes.grad 
        xNew = x .- alpha .* lookAheadGrad + momentum
        prevX = x
        x = xNew

        fCur = f(x)

        logIter!(logger, fCur, x, lookAheadGrad, 1 + lookAheadRes.funcEvals, lookAheadRes.funcEvals, 1, momentums = momentum)

        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, i)
        if converged 
            setConvergenceReason!(logger, reason)
            break
        end
    end
    finalizeLogger!(logger)
    return (
            minimum = x,
            finalValue = fCur,
            logger = logger
        )
end

#TODO implement adaptive version (ADAM and/or Polyak Stepsize?)