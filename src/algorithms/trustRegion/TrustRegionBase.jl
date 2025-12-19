module TrustRegionInterface
    export TrustRegionMethod, solveTrustRegionModel
    abstract type TrustRegionMethod end

    function solveTrustRegionModel(trustRegionMethod::TrustRegionMethod, grad, B, r)
    end
end

module TrustRegionBaseModule
    using ..GradientEstimatorInterface: GradientEstimator, gradient, hessian
    using ..QuasiNewtonInterface: QuasiNewtonMethod, updateApproximation
    using LinearAlgebra

    function TrustRegionOpt(f, x0, gradEstimator::GradientEstimator, trustRegionMethod::TrustRegionMethod, quasiNewtonMethod::QuasiNewtonMethod; initialRadius = 1.0, beta = 1, maxTRstep = 10., eta = 0.1, tol = 1e-5, boundaryTol = 1e-7, lim = 100, printIter = false)
        x = x0
        path = [copy(x0)]
        gradients = []
        hessians = []  
        functionValues = [f(x0)]

        n = length(x0)
        B = Matrix{Float64}(beta * I, n, n)  # Initialize as beta x identity
        grad = gradient(gradEstimator, f, x0)
        r = initialRadius
        
        for i in 1:lim
            if norm(grad) < tol
                if printIter
                    println("Trust Region method converged in $i iterations. Final gradient norm: $(norm(grad))")
                end
                return (
                    minimum = x,
                    path = path,
                    gradients = gradients,
                    hessians = hessians,
                    functionValues = functionValues,
                    iterations = length(path) - 1
                )
            end

            m = p -> functionValues[end] + dot(grad, p) + 0.5 * dot(p, B * p)
            trustRegionResult = solveTrustRegionModel(trustRegionMethod, grad, B, r)
            p = trustRegionResult.p
            
            rho = (f(x) - f(x + p)) / (m(zeros(n)) - m(p))

            if rho < 0.25
                r  = r / 4
            else
                if rho > 0.75 && abs(norm(p) - r) < boundaryTol
                    r = min(2r, maxTRstep) 
                end
            end

            if rho > eta
                xNew = x + p
                gradNew = gradient(gradEstimator, f, xNew)
                s = xNew - x
                y = gradNew - grad
                B = updateApproximation(quasiNewtonMethod, B, s, y)
                x = xNew
                grad = gradNew
                push!(path, x)
                push!(gradients, grad)
                push!(hessians, B)
                push!(functionValues, f(x))
                
            end
        end
        @warn "Trust Region Method maximum iteration reached"
        return (
            minimum = x,
            path = path,
            gradients = gradients,
            hessians = hessians,
            functionValues = functionValues,
            iterations = lim
        )
    end
end