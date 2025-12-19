# module FletcherReevesModule
#     using ..GradientEstimatorInterface: GradientEstimator, gradient, hessian
#     using ..NonlinearCGInterface: NonlinearCGMethod
#     import ..NonlinearCGInterface: calculateBeta
#     using LinearAlgebra
#     export FletcherReeves

"""
Fletcher-Reeves Nonlinear Conjugate Gradient Method

Reference(s)

"""

struct FletcherReeves <:NonlinearCGMethod
end

"""
calculateBeta
    Concrete implementation of Fletcher Reeves method
    Follows interface from NonlinearCGBaseModule

Input
    method (FletcherReeves) - NonlinearCGMethod struct, unused in the method but required for julia to perform multiple dispatch
    gradNew (Matrix) - Matrix containing gradients at next point
    grad (Matrix) - Matrix containing gradients at current point
    direction (Vector) - Current direction
"""

function calculateBeta(method::FletcherReeves, gradNew, grad, direction)
    return (gradNew' * gradNew) / (grad' * grad)    
end
