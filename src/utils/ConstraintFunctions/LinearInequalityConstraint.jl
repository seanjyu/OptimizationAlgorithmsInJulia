struct LinearInequalityConstraint <: Constraint
    a::Vector{Float64}
    b::Float64
    tol::Float64

    function LinearInequalityConstraint(a::Vector{Float64}, b::Float64; tol::Float64 = 1e-8)
        norm(a) > tol || throw(ArgumentError("Constraint vector 'a' cannot be zero"))
        new(a, b, tol)
    end
end

function project(constraint::LinearInequalityConstraint, x)
    a, b = constraint.a, constraint.b
    residual = dot(a, x) - b
    if residual <= 0
        return x  # already feasible
    else
        # project onto the boundary hyperplane aᵀx = b
        return x - (residual / dot(a, a)) * a
    end
end

function isFeasible(constraint::LinearInequalityConstraint, x)
    return dot(constraint.a, x) <= constraint.b + constraint.tol
end

function violation(constraint::LinearInequalityConstraint, x)
    return max(0.0, dot(constraint.a, x) - constraint.b)
end

function gradient(constraint::LinearInequalityConstraint, x)
    return constraint.a
end

function isInequality(constraint::LinearInequalityConstraint)
    return true
end
