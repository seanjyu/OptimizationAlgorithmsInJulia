# module LinearEqualityConstraintModule
    # using ..ConstraintInterface: Constraint, project, isFeasible, jacobian
    # using LinearAlgebra
    
    # export LinearEqualityConstraint, MultipleLinearEqualityConstraint, CombineLinearEqualityConstraints
    
struct LinearEqualityConstraint <: Constraint
    a::Vector{Float64}
    b::Float64
    tol::Float64

    LinearEqualityConstraint(a::Vector{Float64}, b::Float64; tol = 1e-8) = new(a, b, tol) 
end

function project(parameters::LinearEqualityConstraint, x)
    r = residual(parameters, x)[1]
    norm_sq = dot(a, a)
    return x - (r / norm_sq) * a
end

function isFeasible(parameters::LinearEqualityConstraint, x)
    return abs(dot(parameters.a, x) - parameters.b) < parameters.tol
end

function violation(parameters::LinearEqualityConstraint, x)
    return abs(dot(parameters.a, x) - parameters.b)
end

function gradient(parameters::LinearEqualityConstraint, x)
    return parameters.a
end

function constraintDimension(c::LinearEqualityConstraint)
    return 1
end

function residual(c::LinearEqualityConstraint, x)
    return [dot(c.a, x) - c.b]
end


struct MultipleLinearEqualityConstraint <: Constraint
    A::Matrix{Float64}  # m × n matrix (m constraints, n variables)
    b::Vector{Float64}  # m-vector
    tol::Float64
    
    function MultipleLinearEqualityConstraint(A::Matrix{Float64}, b::Vector{Float64}; tol::Float64 = 1e-8)
        # check dimensions
        size(A, 1) == length(b) || throw(DimensionMismatch(
            "A has $(size(A,1)) rows but b has $(length(b)) elements"))


        # check rank
        m, _ = size(A)

        rank_A = rank(A; atol=tol)
        
        if rank_A < m
            @warn "Constraint matrix A has rank $rank_A but $m rows. " *
                    "System has $(m - rank_A) redundant constraint(s). " *
                    "Consider removing redundant constraints for better numerical stability."
        end
        
        # Check consistency: if Ax=b has no solution, catch it early
        # Form augmented matrix [A | b] and check rank
        augmented = hcat(A, b)
        rank_augmented = rank(augmented; atol=tol)
        
        if rank_augmented > rank_A
            throw(ArgumentError(
                "Inconsistent constraint system: rank([A|b]) = $rank_augmented > rank(A) = $rank_A. " *
                "The system Ax = b has no solution."))
        end

        new(A, b, tol)
    end
end

function project(parameters::MultipleLinearEqualityConstraint, x)
    # residual = parameters.A * x - parameters.b
    r = residual(parameters, x)
    lambda = (parameters.A * parameters.A') \ r
    return x - parameters.A' * lambda
end

function isFeasible(parameters::MultipleLinearEqualityConstraint, x)
    r = residual(parameters, x)
    return all(abs.(r) .< parameters.tol)
end

function violation(parameters::MultipleLinearEqualityConstraint, x)
    return norm(parameters.A * x - parameters.b)
end

function jacobian(parameters::MultipleLinearEqualityConstraint, x)
    return parameters.A 
end

function CombineLinearEqualityConstraints(constraints::Vector{LinearEqualityConstraint})
    isempty(constraints) && throw(ArgumentError("Cannot combine empty constraint vector"))
    
    A = stack(c.a for c in constraints; dims=1)
    b = [c.b for c in constraints]
    tol = minimum(c.tol for c in constraints)
    
    return MultipleLinearEqualityConstraint(A, b; tol=tol)
end

function isInequality(constraints::LinearEqualityConstraint)
    return false
end

function isInequality(constraints::MultipleLinearEqualityConstraint)
    return false
end

function residual(constraints::MultipleLinearEqualityConstraint, x)
    return constraints.A * x - constraints.b
end

function constraintDimension(constraints::MultipleLinearEqualityConstraint)
    return size(constraints.A, 1)
end