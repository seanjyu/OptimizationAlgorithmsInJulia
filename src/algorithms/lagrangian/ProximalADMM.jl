struct ProximalADMM <: ADMMVariant
    sigmaX::Float64
    sigmaZ::Float64
end

ProximalADMM(; sigmaX=0.1, sigmaZ=0.1) = ProximalADMM(sigmaX, sigmaZ)

function solveXSubproblem(variant::ProximalADMM, f, A, B, c, x, z, lambda, rho, xSolver)
    u = lambda / rho
    xRef = copy(x)
    xObj(y) = f(y) + (rho / 2) * sum(abs2, A * y + B * z - c + u) + 
              (variant.sigmaX / 2) * sum(abs2, y - xRef)
    return solveUnconstrainedOpt(xObj, x, xSolver).minimum
end

function solveZSubproblem(variant::ProximalADMM, g, A, B, c, x, z, lambda, rho, zSolver)
    u = lambda / rho
    zRef = copy(z)
    zObj(w) = g(w) + (rho / 2) * sum(abs2, A * x + B * w - c + u) + 
              (variant.sigmaZ / 2) * sum(abs2, w - zRef)
    return solveUnconstrainedOpt(zObj, z, zSolver).minimum
end