struct Dykstra <: ProjectionAlgorithm end

function project(composite::CompositeConstraint{Dykstra}, x)
    m = length(composite.constraints)
    y = copy(x)
    increments = [zero(x) for _ in 1:m]

    for _ in 1:composite.max_iter
        y_old = copy(y)
        for (i, c) in enumerate(composite.constraints)
            z = y + increments[i]
            p = project(c, z)
            increments[i] = z - p
            y = p
        end
        norm(y - y_old) < composite.tol && break
    end
    return y
end