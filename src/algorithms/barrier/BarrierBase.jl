abstract type BarrierMethod end

function checkAllInequalityConstraints(constraints::Constraint)
    return isInequality(constraints)
end

function checkAllInequalityConstraints(constraints::CompositeConstraint)
    return all(isInequality, constraints.constraints)
end

function BarrierOpt(f, 
                    x0, 
                    unconstrainedOptSolver::UnconstrainedOptMethod, 
                    constraints::Constraint, 
                    barrierMethod::BarrierMethod, 
                    criteria::ConvergenceCriteria; 
                    rho = 1.1, 
                    lim = 100, 
                    barrierCoefficientStart=1.0,
                    barrierCoefficientMin=1e-10, 
                    track = true)

    # check if constraints are all inequalities
    if !checkAllInequalityConstraints(constraints)
        error("Barrier method requires all constraints to be inequalities")
    end

    # check if starting point is feasible
    if !isFeasible(constraints, x0)
        error("Barrier method requires a feasible starting point")
    end

    x = x0
    fCur = f(x0)
    mu = barrierCoefficientStart

    algorithmData = track ? 
        AlgorithmData(lim; barrierCoefficient=Float64, distToBoundary=Float64) : 
        NoAlgorithmData()

    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim
        fObjectiveBarrier(y) = f(y) + mu * barrierValue(barrierMethod, violation(constraints, y))
        
        unconstrainedRes = solveUnconstrainedOpt(fObjectiveBarrier, x, unconstrainedOptSolver)
        xNew = unconstrainedRes.minimum
        
        # Check solver didn't leave feasible region
        if !isFeasible(constraints, xNew)
            @warn "Barrier solver left feasible region at iteration $i"
            break
        end
        
        fNew = f(xNew)
        viol = violation(constraints, xNew)
        distToBoundary = -maximum(viol)  # distance to nearest constraint

        logIter!(logger, fNew, xNew, zeros(length(x)), 
                unconstrainedRes.logger.functionEvals, 
                unconstrainedRes.logger.gradEstFunctionEvals, 
                unconstrainedRes.logger.gradientEvals;
                barrierCoefficient=mu, distToBoundary=distToBoundary)

        converged, reason = CheckConvergence(criteria, zeros(length(x)), xNew, x, fNew, fCur, i)
        
        if converged
            x, fCur = xNew, fNew
            setConvergenceReason!(logger, reason)
            break
        end

        mu = max(rho * mu, barrierCoefficientMin)
        x, fCur = xNew, fNew
    end

    finalizeLogger!(logger)

    return (minimum=x, finalValue=fCur, logger=logger)
end