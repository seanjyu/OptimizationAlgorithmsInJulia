struct POCS <: ProjectionAlgorithm end

function project(composite::CompositeConstraint{POCS}, x)
    y = copy(x)
    for _ in 1:composite.max_iter
        y_old = copy(y)
        for c in composite.constraints
            y = project(c, y)
        end
        norm(y - y_old) < composite.tol && break
    end
    return y
end