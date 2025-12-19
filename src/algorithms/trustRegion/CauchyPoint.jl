module CauchyPointModule
    using ..TrustRegionInterface: TrustRegionMethod
    import ..TrustRegionInterface: solveTrustRegionModel
    export CauchyPoint

    struct CauchyPoint <: TrustRegionMethod 
        tol::Float64
        CauchyPoint(;tol = 1e-6) = new(tol)
    end

    function solveTrustRegionModel(parameters::CauchyPoint, grad, B, r)
        gradTBgrad = dot(grad, B * grad)

        if gradTBgrad < parameters.tol
            tau =  1.0
        else
            tau = min(1.0, norm(grad)^3 / (r * gradTBgrad))
        end
        # return -τ * (Δ_k / norm(g_k)) * g_k
        return - tau * (r / norm(grad)) * grad
    end
end