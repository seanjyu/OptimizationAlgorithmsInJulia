


function ProjectedGradientDescent(f, x0, gradEstimator::GradientEstimator, constraints::Constraint, criteria::ConvergenceCriteria; alpha=0.1, tol=1e-5, lim=100, track = true)
    x = x0
    fCur = f(x0)

    logger = initLogger(track, x0, fCur, lim, algorithmData = NoAlgorithmData())

    for i in 1:lim

        gradRes = gradient(gradEstimator, f, x)
        grad = gradRes.grad

        xOld = copy(x)
        fOld = fCur
         
        # perform gradient descent step
        x = x .- alpha .* grad
        
        # perform projection
        x = project(constraints, x)

        fCur = f(x)  

        #TODO add stepsize to logIter
        # logIter!(logger, fCur, x, grad, x,  1 + gradRes.funcEvals, gradRes.funcEvals, 1)

        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, 0)
        if converged 
            setConvergenceReason!(logger, reason)
            break
        end

        # if norm(grad) < tol; break; end
    end
    # return (
    #         minimum = x,
    #         path = path,
    #         gradients = gradients,
    #         functionValues = functionValues,
    #         iterations = length(path) - 1
        # )
    finalizeLogger!(logger)
    return (
            minimum = x,
            finalValue = fCur,
            logger = logger
        )
end