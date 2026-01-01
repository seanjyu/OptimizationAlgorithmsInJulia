"""
Gradient Descent
"""

function GradientDescent(f, x0, gradEstimator::GradientEstimator, criteria::ConvergenceCriteria; alpha=0.1, tol=1e-5, lim=100, track = true)

    x = copy(x0)
    fCur = f(x0)

    logger = initLogger(track, x0, fCur, lim, algorithmData = NoAlgorithmData())

    for i in 1:lim
        gradRes = gradient(gradEstimator, f, x)
        grad = gradRes.grad

        xOld = copy(x)
        fOld = fCur

        x = x .- alpha .* grad
        
        fCur = f(x)

        logIter!(logger, fCur, x, grad, 1 + gradRes.funcEvals, gradRes.funcEvals, 1)

        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, 0)
        
        if converged 
            setConvergenceReason!(logger, reason)
            break
        end
    end
    finalizeLogger!(logger)
    return (
            minimumPoint = x,
            finalValue = fCur,
            logger = logger
        )
end