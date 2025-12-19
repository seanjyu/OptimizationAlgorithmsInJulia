abstract type ADMMVariant end

"""
Build and solve the x-subproblem
"""
function solveXSubproblem(variant::ADMMVariant, f, A, B, c, x, z, λ, rho, xSolver)
    error("solveXSubproblem not implemented for $(typeof(variant))")
end

"""
Build and solve the z-subproblem 
"""
function solveZSubproblem(variant::ADMMVariant, g, A, B, c, x, z, λ, rho, zSolver)
    error("solveZSubproblem not implemented for $(typeof(variant))")
end

"""
Compute primal residual: Ax + Bz - c
"""
function primalResidual(A, B, c, x, z)
    return A * x + B * z - c
end

"""
Compute dual residual: ρAᵀB(zNew - zOld)
"""
function dualResidual(A, B, rho, zNew, zOld)
    return rho * A' * B * (zNew - zOld)
end

function ADMMOpt(f, g, x0, z0, A::Matrix, B::Matrix, c::Vector,
                 variant::ADMMVariant, criteria::ConvergenceCriteria;
                 xSolver::Union{UnconstrainedOptMethod, Nothing}=nothing,
                 zSolver::Union{UnconstrainedOptMethod, Nothing}=nothing,
                 rho=1.0, rhoMax=1000.0, rhoMin=1e-6,
                 adaptRho=true, muAdapt=10.0, tauAdapt=2.0,
                 primTol=1e-6, dualTol=1e-6,
                 lim=100, track=false)
    
    x, z = copy(x0), copy(z0)
    lambda = zeros(length(c))
    fCur = f(x) + g(z)
    
    algorithmData = track ? 
        AlgorithmData(lim; primalResidual=Float64, dualResidual=Float64, 
                      penaltyCoefficient=Float64) : 
        NoAlgorithmData()
    logger = initLogger(track, x0, fCur, lim; algorithmData=algorithmData)

    for i in 1:lim
        zOld = copy(z)
        
        # x-update
        xNew = solveXSubproblem(variant, f, A, B, c, x, z, lambda, rho, xSolver)
        
        # z-update
        zNew = solveZSubproblem(variant, g, A, B, c, xNew, z, lambda, rho, zSolver)
        
        # Residuals
        primRes = primalResidual(A, B, c, xNew, zNew)
        dualRes = dualResidual(A, B, rho, zNew, zOld)
        primResNorm = norm(primRes)
        dualResNorm = norm(dualRes)
        
        # Dual update
        lambdaNew = lambda + rho * primRes
        
        fNew = f(xNew) + g(zNew)

        logIter!(logger, fNew, xNew, zeros(length(x)), 0.0, 0, 0, 0;
                 primalResidual=primResNorm, dualResidual=dualResNorm, 
                 penaltyCoefficient=rho)

        #TODO use two convergence checkers?
        # Convergence check
        if primResNorm < primTol && dualResNorm < dualTol
            x, z, lambda = xNew, zNew, lambdaNew
            setConvergenceReason!(logger, "Primal and dual residuals below tolerance")
            break
        end

        # Adaptive penalty (Boyd et al. heuristic)
        if adaptRho
            if primResNorm > muAdapt * dualResNorm
                rho = min(tauAdapt * rho, rhoMax)
                lambdaNew = lambdaNew / tauAdapt  # rescale dual variable
            elseif dualResNorm > muAdapt * primResNorm
                rho = max(rho / tauAdapt, rhoMin)
                lambdaNew = lambdaNew * tauAdapt
            end
        end

        x, z, lambda, fCur = xNew, zNew, lambdaNew, fNew
    end

    finalizeLogger!(logger)
    
    return (
        x = x,
        z = z,
        minimum = f(x) + g(z),
        dualVariables = lambda,
        logger = logger
    )
end