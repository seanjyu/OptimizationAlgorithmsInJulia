struct LinearizedADMM <: ADMMVariant
    gradf::Function
    gradg::Function
    tauX::Union{Float64, Nothing}
    tauZ::Union{Float64, Nothing}
end

LinearizedADMM(gradf, gradg; tauX=nothing, tauZ=nothing) = 
    LinearizedADMM(gradf, gradg, tauX, tauZ)

function solveXSubproblem(variant::LinearizedADMM, f, A, B, c, x, z, lambda, rho, xSolver)
    # Step size: must be < 1/(ρ||AᵀA||) for convergence
    tau = variant.tauX === nothing ? 0.9 / (rho * opnorm(A)^2) : variant.tauX
    grad = variant.gradf(x) + A' * lambda + rho * A' * (A * x + B * z - c)
    return x - tau * grad
end

function solveZSubproblem(variant::LinearizedADMM, g, A, B, c, x, z, lambda, rho, zSolver)
    tau = variant.tauZ === nothing ? 0.9 / (rho * opnorm(B)^2) : variant.tauZ
    grad = variant.gradg(z) + B' * lambda + rho * B' * (A * x + B * z - c)
    return z - tau * grad
end