
"""
Dai Yuan Nonlinear Conjugate Gradient Method

Reference(s)
"""


struct DaiYuan <:NonlinearCGMethod
end

"""
calculateBeta
    Concrete implementation of Dai Yuan Method
    Follows interface from NonlinearCGBaseModule

Input
    method (NonlinearCGMethod) - NonlinearCGMethod struct, unused in the method but required for julia to perform multiple dispatch
    gradNew (Matrix) - Matrix containing gradients at next point
    grad (Matrix) - Matrix containing gradients at current point
    direction (Vector) - Current direction
"""

function calculateBeta(method::DaiYuan, gradNew, grad, direction)
    return (gradNew' * gradNew) / (direction' * (gradNew - grad))    
end