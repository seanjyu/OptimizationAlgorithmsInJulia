function DualAscent(f, x0, unconstrainedOptSolver::UnconstrainedOptMethod,
                    A::Matrix, b::Vector, criteria::ConvergenceCriteria;
                    alpha=0.1, lim=100, track=false)
    
    x = x0
    lambda = zeros(length(b))  # dual variables
    fCur = f(x0)

    algorithmData = track ? 
        AlgorithmData(lim; dualNorm=Float64, primalResidual=Float64) : 
        NoAlgorithmData()
    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim
        # x-minimization: min_x L(x, lambda) = f(x) + dot(lambda, (Ax - b))
        lagrangian(y) = f(y) + dot(lambda, A * y - b)
        
        unconstrainedRes = solveUnconstrainedOpt(lagrangian, x, unconstrainedOptSolver)
        xNew = unconstrainedRes.minimum
        fNew = f(xNew)
        
        # Dual update: gradient ascent on dual function
        primalResidual = A * xNew - b
        lambdaNew = lambda + alpha * primalResidual

        logIter!(logger, fNew, xNew, zeros(length(x)), 0.0, 0, 0, 0;
                 dualNorm=norm(lambdaNew), primalResidual=norm(primalResidual))

        converged, reason = CheckConvergence(criteria, zeros(length(x)), xNew, x, fNew, fCur, i)
        
        if converged && norm(primalResidual) < 1e-6
            x, fCur, lambda = xNew, fNew, lambdaNew
            setConvergenceReason!(logger, reason)
            break
        end

        x, fCur, lambda = xNew, fNew, lambdaNew
    end

    finalizeLogger!(logger)
    return (minimum=x, finalValue=fCur, dualVariables=lambda, logger=logger)
end