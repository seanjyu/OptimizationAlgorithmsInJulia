"""
Polak Ribiere Nonlinear Conjugate Gradient Method

Reference(s)

"""
struct PolakRibiere <:NonlinearCGMethod
end

"""
calculateBeta
    Concrete implementation of Polak Ribiere Method
    Follows interface from NonlinearCGBaseModule

Input
    method (NonlinearCGMethod) - NonlinearCGMethod struct, unused in the method but required for julia to perform multiple dispatch
    gradNew (Matrix) - Matrix containing gradients at next point
    grad (Matrix) - Matrix containing gradients at current point
    direction (Vector) - Current direction
"""
function calculateBeta(method::PolakRibiere, gradNew, grad, direction)
    return (gradNew' * (gradNew - grad)) / (grad' * grad)  
end