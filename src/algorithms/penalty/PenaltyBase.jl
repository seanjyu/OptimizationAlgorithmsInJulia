abstract type PenaltyMethod end

function penaltyValue(parameters::PenaltyMethod, violations)
end

function PenaltyOpt(f, 
                    x0, 
                    unconstrainedOptSolver::UnconstrainedOptMethod, 
                    constraints::Constraint, 
                    penaltyMethod::PenaltyMethod, 
                    criteria::ConvergenceCriteria; 
                    rho = 1.1, 
                    lim = 100, 
                    penaltyCoefficientStart = 1, 
                    penaltyCoefficientMax = 100, 
                    track = true)
    x = x0
    fCur = f(x0)
    mu = penaltyCoefficientStart

    algorithmData = track ? 
        AlgorithmData(lim; penaltyCoefficient=Float64, constraintViolation=Float64) : 
        NoAlgorithmData()

    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim

        # build objective function with penaltyCoefficient
        fObjectivePenalty(y) = f(y) + mu * penaltyValue(penaltyMethod, violation(constraints, y))
        
        unconstrainedRes = solveUnconstrainedOpt(fObjectivePenalty, x, unconstrainedOptSolver)
        xNew = unconstrainedRes.minimum
        fNew = f(xNew)

        viol = violation(constraints, xNew)

        logIter!(logger, fNew, xNew, zeros(length(x)), 
                0.0, # no value for step size (?)
                unconstrainedRes.logger.functionEvals, 
                unconstrainedRes.logger.gradEstFunctionEvals, 
                unconstrainedRes.logger.gradientEvals;
                penaltyCoefficient=mu, constraintViolation=norm(viol))

        converged, reason = CheckConvergence(criteria, zeros(length(x)), xNew, x, fNew, fCur, i)
        
        if isFeasible(constraints, xNew) && converged
            x, fCur = xNew, fNew
            setConvergenceReason!(logger, reason)
            break
        end

        mu = min(rho * mu, penaltyCoefficientMax)
        x, fCur = xNew, fNew


    end

    finalizeLogger!(logger)

    return (
        minimum = x,
        finalValue = fCur,
        logger = logger
    )

end