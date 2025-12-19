"""
Hestenes Stiefel Nonlinear Conjugate Gradient Method

Reference(s)

"""

struct HestenesStiefel <:NonlinearCGMethod
end

"""
calculateBeta
    Concrete implementation of Hestenes Stiefel Method
    Follows interface from NonlinearCGBaseModule

Input
    method (NonlinearCGMethod) - NonlinearCGMethod struct, unused in the method but required for julia to perform multiple dispatch
    gradNew (Matrix) - Matrix containing gradients at next point
    grad (Matrix) - Matrix containing gradients at current point
    direction (Vector) - Current direction
"""

function calculateBeta(method::HestenesStiefel, gradNew, grad, direction)
    return (gradNew' * (gradNew - grad)) / ((gradNew - grad)' * direction)    
end
# end