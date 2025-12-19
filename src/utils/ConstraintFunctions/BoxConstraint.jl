# module BoxConstraintModule
#     using ..ConstraintInterface: Constraint, project, isFeasible, jacobian
#     using LinearAlgebra

#     export BoxConstraint

#     struct BoxConstraint <: Constraint
#         lower::Vector{Float64}
#         upper::Vector{Float64}
#         tol::Float64
#     end
# end

struct BoxConstraint <: Constraint
    lower::Vector{Float64}
    upper::Vector{Float64}
    tol::Float64

    function BoxConstraint(lower::Vector{Float64}, upper::Vector{Float64}; tol::Float64 = 1e-8)
        length(lower) == length(upper) || throw(DimensionMismatch(
            "lower has $(length(lower)) elements but upper has $(length(upper)) elements"))
        
        all(lower .<= upper .+ tol) || throw(ArgumentError(
            "lower bounds must be <= upper bounds"))
        
        new(lower, upper, tol)
    end
end

function project(constraint::BoxConstraint, x)
    # Clamp each element to [lower, upper]
    y = clamp.(x, constraint.lower, constraint.upper)
    # return y, norm(y - x)
    return y
end

function isFeasible(constraint::BoxConstraint, x)
    return all(constraint.lower .- constraint.tol .<= x .<= constraint.upper .+ constraint.tol)
end

function violation(constraint::BoxConstraint, x)
    lower_violations = max.(constraint.lower .- x, 0.0)
    upper_violations = max.(x .- constraint.upper, 0.0)
    return norm([lower_violations; upper_violations])
end

function jacobian(constraint::BoxConstraint, x)
    # Jacobian is complex for box constraints since different constraints
    # are active at different points. Return indicator of active constraints.
    n = length(x)
    active = zeros(0, n)
    
    for i in 1:n
        if x[i] <= constraint.lower[i] + constraint.tol
            row = zeros(n)
            row[i] = -1.0
            active = vcat(active, row')
        elseif x[i] >= constraint.upper[i] - constraint.tol
            row = zeros(n)
            row[i] = 1.0
            active = vcat(active, row')
        end
    end
    
    return active
end

function isInequality(constraint::BoxConstraint)
    return true
end