module UnivariateForwardDifferenceModule
    using ..GradientEstimatorInterface: GradientEstimator
    import ..GradientEstimatorInterface: gradient, hessian
    export UnivariateForwardDifference
    
    """
    Module containing univariate forward difference algorithms
    Currently only the 3 point central difference is implemented, in the future will implement more points
    """ 

    struct UnivariateForwardDifference <: GradientEstimator
        h::Float64
    end
    
    function gradient(est::UnivariateForwardDifference, f, x::Number) 
        return (f(x + est.h) - f(x)) / est.h
    end

    function hessian(est::UnivariateForwardDifference, f, x::Number)
        return (f(x + est.h) - 2*f(x) + f(x - est.h)) / est.h^2
    end
end

module MultivariateForwardDifferenceModule

    using ..GradientEstimatorInterface: GradientEstimator
    import ..GradientEstimatorInterface: gradient, hessian
    export MultivariateForwardDifference
    
    """
    Module containing multivariate forward difference algorithms
    Currently only the 3 point central difference is implemented (4 point for mixed partial derivatives), in the future will implement more points
    """

    struct MultivariateForwardDifference <: GradientEstimator
        h::Float64
    end

    function gradient(est::MultivariateForwardDifference, f, x::Vector{<:Number})
    n = length(x)
    grad = zeros(eltype(x), n)
    fx = f(x)

    for i in 1:n    
        x_forward = copy(x)
        x_forward[i] += est.h
        grad[i] = (f(x_forward) - fx) / est.h
    end

    return grad
    end

    function hessian(est::MultivariateForwardDifference, f, x::Vector{<:Number})
    n = length(x)
    H = zeros(eltype(x), n, n)
    fx = f(x)
    
    # Diagonal elements: H[i,i]
    for i in 1:n
        x_forward = copy(x)
        x_forward[i] += est.h
        x_backward = copy(x)
        x_backward[i] -= est.h
        H[i,i] = (f(x_forward) - 2*fx + f(x_backward)) / est.h^2
    end
    
    # Off-diagonal elements: H[i,j] where i < j
    for i in 1:n
        for j in (i+1):n
            x_pp = copy(x)  # + +
            x_pp[i] += est.h
            x_pp[j] += est.h
            
            x_pm = copy(x)  # + -
            x_pm[i] += est.h
            x_pm[j] -= est.h
            
            x_mp = copy(x)  # - +
            x_mp[i] -= est.h
            x_mp[j] += est.h
            
            x_mm = copy(x)  # - -
            x_mm[i] -= est.h
            x_mm[j] -= est.h
            
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * est.h^2)
            H[j,i] = H[i,j]  
        end
    end
    
    return H
    end    
end

