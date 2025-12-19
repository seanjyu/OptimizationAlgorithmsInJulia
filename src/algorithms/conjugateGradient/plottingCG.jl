
include("../../utils/GradientEstimators/GradientEstimatorType.jl")
include("../../utils/visual/VisualizationFunctions.jl")
include("../../utils/TestFunctions.jl")
include("ConjugateGradient.jl")

using .GradientEstimatorInterface: GradientEstimator, gradient
using .MultivariatePlottingModule
using .MultivariateQuadraticFunctionModule
using .ConjugateGradientModule
using Plots

A = [5.0 1.0; 
     1.0 2.0]
b = [-3.0, 4.0]
c = 5.0


qf = MultivariateQuadraticFunction(A, b, c)

startingPoint = [10.0, -8.0]

res = ConjugateGradient(A, -b, startingPoint)
println(res)
println(trueMinimum(qf))

# Plot 
p = plot2DContourFunction(qf, -10, 10, -8, 5, 0.1, fill=false)

scatter!([startingPoint[1]],[startingPoint[2]],
    label = "",
    markersize=6,
    color=:green)


