module DogLegModule
    using ..TrustRegionInterface: TrustRegionMethod
    import ..TrustRegionInterface: solveTrustRegionModel
    export DogLeg

    struct DogLeg <: TrustRegionMethod 
        tol::Float64
        DogLeg(;tol = 1e-6) = new(tol)
    end

    function solveTrustRegionModel(parameters::CauchyPoint, grad, B, r)
        # compute Newton step
        pB = -B \ grad
        
        # return Newton step if within radius
        if norm(p) <= r
            return p  
        end
        
        # compute steepest descent step
        pU = -(dot(grad, grad) / dot(grad, B * grad)) * grad
        
        # return if larger than radius
        if norm(pU) >= r
            return -(r / norm(grad)) * grad
        end
        
        # Find intersection on second leg (p_  U to p_B)
        # Solve ||pU + tau(pB - pU)||² = r²
        a = norm(pB - pU)^2
        b = 2 * dot(pU, p_B - pU)
        c = norm(pU)^2 - r^2
        
        tau = (-b + sqrt(b^2 - 4*a*c)) / (2*a)
        
        return pU + tau * (pB - pU)
    end
end