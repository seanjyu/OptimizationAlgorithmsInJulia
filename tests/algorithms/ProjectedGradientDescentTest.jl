using Test
include("../../src/OptAlgos.jl")
using .OptAlgos

# Find point in unit simplex closest to a target point

using LinearAlgebra

# Target point (outside the simplex)
# target = [0.8, 0.6, 0.5]
target = [2.0, 3.0, -1.0]  # way outside simplex

# Objective: minimize squared distance to target
f(x) = sum((x .- target).^2)

# Constraints: x ≥ 0 and sum(x) = 1
box = BoxConstraint([0.0, 0.0, 0.0], [Inf, Inf, Inf])
equality = LinearEqualityConstraint([1.0, 1.0, 1.0], 1.0)

# Without constraints: minimum is [2.0, 3.0, -1.0]
x_unconstrained = target
println("Unconstrained: ", x_unconstrained)
println("Value: ", f(x_unconstrained))

# Create composite constraint with Dykstra's algorithm
constraints = CompositeConstraint([box, equality]; method=Dykstra)

# Check if starting point is feasible
x0 = [0.3, 0.3, 0.4]
println("x0 feasible: ", isFeasible(constraints, x0))
println("x0 violation: ", violation(constraints, x0))

# Project an infeasible point
x_infeasible = [0.8, 0.6, 0.5]
x_projected = project(constraints, x_infeasible)
println("\nProjected point: ", x_projected)
println("Sum: ", sum(x_projected))  # should be 1.0
println("All positive: ", all(x_projected .>= 0))
println("Is feasible: ", isFeasible(constraints, x_projected))

GradientEstimator = FiniteDifferenceMultivariate{5,:central}(3e-6)

c = ConvergenceCriteria()
# Run projected gradient descent
result = ProjectedGradientDescent(
    f, 
    x0, 
    GradientEstimator,
    constraints,
    c;
    alpha = 0.1,
    lim = 1000
)

println("\nOptimization result:")
println("Optimal point: ", result.minimum)
println("Distance to target: ", sqrt(result.finalValue))
println("Sum constraint: ", sum(result.minimum))
println("Is feasible: ", isFeasible(constraints, result.minimum))