abstract type ProjectionAlgorithm end


struct CompositeConstraint{M<:ProjectionAlgorithm} <: Constraint
    constraints::Vector{<:Constraint}
    tol::Float64
    max_iter::Int

    function CompositeConstraint{M}(constraints::Vector{<:Constraint}; 
                                     tol::Float64 = 1e-8, 
                                     max_iter::Int = 1000) where M
        isempty(constraints) && throw(ArgumentError("Cannot create empty composite constraint"))
        new{M}(constraints, tol, max_iter)
    end
end

# Convenience constructor defaulting to Dykstra
function CompositeConstraint(constraints::Vector{<:Constraint}; 
                              method::Type{<:ProjectionAlgorithm} = Dykstra,
                              tol::Float64 = 1e-8, 
                              max_iter::Int = 1000)
    CompositeConstraint{method}(constraints; tol=tol, max_iter=max_iter)
end

function isFeasible(composite::CompositeConstraint, x)
    return all(isFeasible(c, x) for c in composite.constraints)
end

# function violation(composite::CompositeConstraint, x)
#     return sqrt(sum(violation(c, x)^2 for c in composite.constraints))
# end

function violation(composite::CompositeConstraint, x)
    return [violation(c, x) for c in composite.constraints]
end

function residual(composite::CompositeConstraint, x)
    return vcat((residual(c, x) for c in composite.constraints)...)
end

function jacobian(composite::CompositeConstraint, x)
    return vcat([jacobian(c, x) for c in composite.constraints]...)
end