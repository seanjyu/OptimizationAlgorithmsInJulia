struct BasicADMM <: ADMMVariant end

function solveXSubproblem(::BasicADMM, f, A, B, c, x, z, lambda, rho, xSolver)
    u = lambda / rho
    xObj(y) = f(y) + (rho / 2) * sum(abs2, A * y + B * z - c + u)
    return solveUnconstrainedOpt(xObj, x, xSolver).minimum
end

function solveZSubproblem(::BasicADMM, g, A, B, c, x, z, lambda, rho, zSolver)
    u = lambda / rho
    zObj(w) = g(w) + (rho / 2) * sum(abs2, A * x + B * w - c + u)
    return solveUnconstrainedOpt(zObj, z, zSolver).minimum
end