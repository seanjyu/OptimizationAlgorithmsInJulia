function DualAscent(f, x0, unconstrainedOptSolver::UnconstrainedOptMethod,
                    constraints::Constraint, criteria::ConvergenceCriteria;
                    alpha=0.1, lim=100, track=false)

    if isa(constraints, CompositeConstraint)
        any(isInequality(c) for c in constraints.constraints) &&
            error("DualAscent only supports equality constraints")
    elseif isInequality(constraints)
        error("DualAscent only supports equality constraints")
    end
    
    x = x0
    lambda = zeros(constraintDimension(constraints))
    fCur = f(x0)

    algorithmData = track ? 
        AlgorithmData(lim; dualNorm=Float64, primalResidual=Float64) : 
        NoAlgorithmData()
    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    # TODO implement early stopping?
    for i in 1:lim
        # x-minimization: min_x L(x, lambda) = f(x) + dot(lambda, (Ax - b))
        lagrangian(y) = f(y) + dot(lambda, residual(constraints, y))
        
        unconstrainedRes = solveUnconstrainedOpt(lagrangian, x, unconstrainedOptSolver)
        xNew = unconstrainedRes.minimumPoint
        fNew = f(xNew)
        
        # Dual update: gradient ascent on dual function
        primalResidual = residual(constraints, xNew)
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