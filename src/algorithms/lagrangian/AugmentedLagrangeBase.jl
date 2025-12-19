abstract type AugmentedLagrangianMethod end

"""
Build the augmented Lagrangian subproblem objective
"""
function buildSubproblem(variant::AugmentedLagrangianMethod, f, constraints, x, lambda, rho)
    error("buildSubproblem not implemented for $(typeof(variant))")
end

"""
Update dual variables (most variants use the same rule, but can override)
"""
function updateDual(::AugmentedLagrangianMethod, lambda, rho, constraintResidual)
    return lambda + rho * constraintResidual
end

"""
Update penalty parameter (can override for adaptive strategies)
"""
function updatePenalty(::AugmentedLagrangianMethod, rho, rhoMax, rhoScale, residualOld, residualNew)
    if residualNew > 0.25 * residualOld
        return min(rho * rhoScale, rhoMax)
    end
    return rho
end

struct BasicAugmentedLagrangianMethod <: AugmentedLagrangianMethod end

function buildSubproblem(::BasicAugmentedLagrangianMethod, f, constraints, x, lambda, rho)
    return y -> begin
        c = residual(constraints, y)
        f(y) + dot(lambda, c) + (rho / 2) * sum(abs2, c)
    end
end

function AugmentedLagrangianOpt(f, x0, unconstrainedOptSolver::UnconstrainedOptMethod,
                                 constraints::Constraint, variant::AugmentedLagrangianMethod,
                                 criteria::ConvergenceCriteria;
                                 rho=1.0, rhoMax=1000.0, rhoScale=1.5,
                                 lim=100, feasTol=1e-6, track=true)
    
    x = x0
    fCur = f(x0)
    res = residual(constraints, x0)
    lambda = zeros(length(res))
    
    algorithmData = track ? 
        AlgorithmData(lim; dualNorm=Float64, primalResidual=Float64, penaltyCoefficient=Float64) : 
        NoAlgorithmData()
    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    residualOld = norm(res)

    for i in 1:lim
        # Build subproblem using variant-specific logic
        subproblem = buildSubproblem(variant, f, constraints, x, lambda, rho)
        
        unconstrainedRes = solveUnconstrainedOpt(subproblem, x, unconstrainedOptSolver)
        xNew = unconstrainedRes.minimum
        fNew = f(xNew)
        
        cNew = residual(constraints, xNew)
        residualNew = norm(cNew)
        
        # Dual update (variant can override)
        lambdaNew = updateDual(variant, lambda, rho, cNew)

        logIter!(logger, fNew, xNew, zeros(length(x)), 
                unconstrainedRes.logger.functionEvals, 
                unconstrainedRes.logger.gradEstFunctionEvals, 
                unconstrainedRes.logger.gradientEvals;
                 dualNorm=norm(lambdaNew), primalResidual=residualNew, penaltyCoefficient=rho)

        converged, reason = CheckConvergence(criteria, zeros(length(x)), xNew, x, fNew, fCur, i)
        
        if converged && residualNew < feasTol
            x, fCur, lambda = xNew, fNew, lambdaNew
            setConvergenceReason!(logger, reason)
            break
        end

        # Penalty update (variant can override)
        rho = updatePenalty(variant, rho, rhoMax, rhoScale, residualOld, residualNew)
        
        x, fCur, lambda = xNew, fNew, lambdaNew
        residualOld = residualNew
    end

    finalizeLogger!(logger)
    return (minimum=x, finalValue=fCur, dualVariables=lambda, logger=logger)
end