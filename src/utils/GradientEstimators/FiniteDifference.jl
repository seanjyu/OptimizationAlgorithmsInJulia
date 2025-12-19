"""
FiniteDifferenceUnivariate{N,T}(h::Float64)

    Finite difference estimator for univariate (scalar) functions.

Type Inputs
    N (Int) - Number of points in stencil (2, 3, 5, 7)
    T (Symbol) - Type of difference (:forward, :central, :backward)

Input
    h (Float64) - Step size for finite differences

# Examples
```
# 3-point central difference 
est = FiniteDifferenceUnivariate{3,:central}(1e-5)
result = gradient(est, x -> x^2, 2.0)
# result.grad ≈ 4.0, result.funcEvals = 2

# 5-point central difference 
est = FiniteDifferenceUnivariate{5,:central}(1e-6)
result = hessian(est, x -> x^3, 1.0)
# result.hess ≈ 6.0, result.funcEvals = 5
```
"""
struct FiniteDifferenceUnivariate{N,T} <: GradientEstimator
    h::Float64
    
    function FiniteDifferenceUnivariate{N,T}(h::Float64) where {N,T}
        h > 0 || throw(ArgumentError("Step size h must be positive"))
        
        if T == :forward
            N >= 2 || throw(ArgumentError("Forward difference needs N ≥ 2"))
        elseif T == :central
            N >= 3 && isodd(N) || throw(ArgumentError("Central difference needs odd N ≥ 3"))
        elseif T == :backward
            N >= 2 || throw(ArgumentError("Backward difference needs N ≥ 2"))
        else
            throw(ArgumentError("Type must be :forward, :central, or :backward"))
        end
        
        new{N,T}(h)
    end
end

# 2-point forward difference 
function gradient(est::FiniteDifferenceUnivariate{2,:forward}, f, x::Number)
    h = est.h
    grad = (f(x + h) - f(x)) / h
    return (grad = grad, funcEvals = 2)
end

# 3-point forward difference 
function gradient(est::FiniteDifferenceUnivariate{3,:forward}, f, x::Number)
    h = est.h
    grad = (-3*f(x) + 4*f(x + h) - f(x + 2h)) / (2h)
    return (grad = grad, funcEvals = 3)
end

# 3-point central difference 
function gradient(est::FiniteDifferenceUnivariate{3,:central}, f, x::Number)
    h = est.h
    grad = (f(x + h) - f(x - h)) / (2h)
    return (grad = grad, funcEvals = 2)
end

# 5-point central difference 
function gradient(est::FiniteDifferenceUnivariate{5,:central}, f, x::Number)
    h = est.h
    grad = (-f(x + 2h) + 8*f(x + h) - 8*f(x - h) + f(x - 2h)) / (12h)
    return (grad = grad, funcEvals = 4)
end

# 7-point central difference 
function gradient(est::FiniteDifferenceUnivariate{7,:central}, f, x::Number)
    h = est.h
    grad = (f(x - 3h) - 9*f(x - 2h) + 45*f(x - h) - 
            45*f(x + h) + 9*f(x + 2h) - f(x + 3h)) / (60h)
    return (grad = grad, funcEvals = 6)
end

function hessian(est::FiniteDifferenceUnivariate{3,:forward}, f, x::Number)
    h = est.h
    hess = (f(x) - 2*f(x + h) + f(x + 2h)) / h^2
    return (hess = hess, funcEvals = 3, gradEvals = 0)
end

# 3-point central difference (O(h²))
function hessian(est::FiniteDifferenceUnivariate{3,:central}, f, x::Number)
    h = est.h
    hess = (f(x + h) - 2*f(x) + f(x - h)) / h^2
    return (hess = hess, funcEvals = 3, gradEvals = 0)
end

# 5-point central difference (O(h⁴))
function hessian(est::FiniteDifferenceUnivariate{5,:central}, f, x::Number)
    h = est.h
    hess = (-f(x + 2h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2h)) / (12h^2)
    return (hess = hess, funcEvals = 5, gradEvals = 0)
end

struct FiniteDifferenceMultivariate{N,T} <: GradientEstimator
    h::Float64
    
    function FiniteDifferenceMultivariate{N,T}(h::Float64) where {N,T}
        h > 0 || throw(ArgumentError("Step size h must be positive"))
        
        if T == :forward
            N >= 2 || throw(ArgumentError("Forward difference needs N ≥ 2"))
        elseif T == :central
            N >= 3 && isodd(N) || throw(ArgumentError("Central difference needs odd N ≥ 3"))
        else
            throw(ArgumentError("Type must be :forward or :central for multivariate"))
        end
        
        new{N,T}(h)
    end
end

# 2-point forward difference 
function gradient(est::FiniteDifferenceMultivariate{2,:forward}, f, x::AbstractVector)
    n = length(x)
    grad = similar(x)
    h = est.h
    
    f0 = f(x)
    for i in 1:n
        xPlus = copy(x)
        xPlus[i] += h
        grad[i] = (f(xPlus) - f0) / h
    end
    
    return (grad = grad, funcEvals = n + 1)
end

function gradient(est::FiniteDifferenceMultivariate{3,:central}, f, x::AbstractVector)
    n = length(x)
    grad = similar(x)
    h = est.h
    
    for i in 1:n
        xPlus = copy(x)
        xMinus = copy(x)
        xPlus[i] += h
        xMinus[i] -= h
        grad[i] = (f(xPlus) - f(xMinus)) / (2h)
    end
    
    return (grad = grad, funcEvals = 2n)
end

# 5-point central difference (O(h⁴))
function gradient(est::FiniteDifferenceMultivariate{5,:central}, f, x::AbstractVector)
    n = length(x)
    grad = similar(x)
    h = est.h
    
    for i in 1:n
        x2h = copy(x); x2h[i] += 2h
        xh = copy(x); xh[i] += h
        xmh = copy(x); xmh[i] -= h
        xm2h = copy(x); xm2h[i] -= 2h
        
        grad[i] = (-f(x2h) + 8*f(xh) - 8*f(xmh) + f(xm2h)) / (12h)
    end
    
    return (grad = grad, funcEvals = 4n)
end

# 7-point central difference (O(h⁶))
function gradient(est::FiniteDifferenceMultivariate{7,:central}, f, x::AbstractVector)
    n = length(x)
    grad = similar(x)
    h = est.h
    
    for i in 1:n
        x3h = copy(x); x3h[i] += 3h
        x2h = copy(x); x2h[i] += 2h
        xh = copy(x); xh[i] += h
        xmh = copy(x); xmh[i] -= h
        xm2h = copy(x); xm2h[i] -= 2h
        xm3h = copy(x); xm3h[i] -= 3h
        
        grad[i] = (f(xm3h) - 9*f(xm2h) + 45*f(xmh) - 
                   45*f(xh) + 9*f(x2h) - f(x3h)) / (60h)
    end
    
    return (grad = grad, funcEvals = 6n)
end

# ============================================================================
# Helper Functions for Hessian Mixed Partials
# ============================================================================

"""
Compute mixed partial derivative ∂²f/∂xi∂xj using 3x3 stencil (O(h²)).
"""
function mixedPartial3x3(f, x, i, j, h)
    xPp = copy(x); xPp[i] += h; xPp[j] += h
    xPm = copy(x); xPm[i] += h; xPm[j] -= h
    xmp = copy(x); xmp[i] -= h; xmp[j] += h
    xmm = copy(x); xmm[i] -= h; xmm[j] -= h
    
    result = (f(xPp) - f(xPm) - f(xmp) + f(xmm)) / (4h^2)
    return result, 4
end

function mixedPartial5x5(f, x, i, j, h)
    xp2p2 = copy(x); xp2p2[i] += 2h; xp2p2[j] += 2h
    xp2m2 = copy(x); xp2m2[i] += 2h; xp2m2[j] -= 2h
    xm2p2 = copy(x); xm2p2[i] -= 2h; xm2p2[j] += 2h
    xm2m2 = copy(x); xm2m2[i] -= 2h; xm2m2[j] -= 2h
    
    xp1p1 = copy(x); xp1p1[i] += h; xp1p1[j] += h
    xp1m1 = copy(x); xp1m1[i] += h; xp1m1[j] -= h
    xm1p1 = copy(x); xm1p1[i] -= h; xm1p1[j] += h
    xm1m1 = copy(x); xm1m1[i] -= h; xm1m1[j] -= h
    
    result = (
        -1 * (f(xp2p2) - f(xp2m2) - f(xm2p2) + f(xm2m2)) +
        16 * (f(xp1p1) - f(xp1m1) - f(xm1p1) + f(xm1m1))
    ) / (48h^2)
    
    return result, 8
end

# 3-point central difference Hessian (O(h²))
function hessian(est::FiniteDifferenceMultivariate{3,:central}, f, x::AbstractVector)
    n = length(x)
    H = zeros(n, n)
    h = est.h
    funcEvals = 1
    
    f0 = f(x)
    
    # Diagonal elements: ∂²f/∂xi²
    for i in 1:n
        xPlus = copy(x); xPlus[i] += h
        xMinus = copy(x); xMinus[i] -= h
        H[i, i] = (f(xPlus) - 2*f0 + f(xMinus)) / h^2
        funcEvals += 2
    end
    
    # Off-diagonal elements: ∂²f/∂xi∂xj
    for i in 1:n
        for j in (i+1):n
            H[i, j], evals = mixedPartial3x3(f, x, i, j, h)
            H[j, i] = H[i, j]
            funcEvals += evals
        end
    end
    
    return (hess = H, funcEvals = funcEvals, gradEvals = 0)
end

# 5-point central difference Hessian (O(h⁴))
function hessian(est::FiniteDifferenceMultivariate{5,:central}, f, x::AbstractVector)
    n = length(x)
    H = zeros(n, n)
    h = est.h
    funcEvals = 1
    
    f0 = f(x)
    
    # Diagonal elements: ∂²f/∂xi² with 5-point stencil
    for i in 1:n
        x2h = copy(x); x2h[i] += 2h
        xh = copy(x); xh[i] += h
        xmh = copy(x); xmh[i] -= h
        xm2h = copy(x); xm2h[i] -= 2h
        
        H[i, i] = (-f(x2h) + 16*f(xh) - 30*f0 + 16*f(xmh) - f(xm2h)) / (12h^2)
        funcEvals += 4
    end
    
    # Off-diagonal elements: ∂²f/∂xi∂xj with 5x5 stencil
    for i in 1:n
        for j in (i+1):n
            H[i, j], evals = mixedPartial5x5(f, x, i, j, h)
            H[j, i] = H[i, j]
            funcEvals += evals
        end
    end
    
    return (hess = H, funcEvals = funcEvals, gradEvals = 0)
end