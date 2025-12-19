struct LogBarrierALM <: AugmentedLagrangianMethod
    mu::Float64  # barrier parameter
end

LogBarrierALM(; mu=0.1) = LogBarrierALM(mu)

function buildSubproblem(variant::LogBarrierALM, f, constraints, x, lambda, rho)
    return y -> begin
        c = residual(constraints, y)
        barrier = all(c .< 0) ? -variant.mu * sum(log.(-c)) : Inf
        f(y) + dot(λ, c) + (rho / 2) * sum(abs2, c) + barrier
    end
end