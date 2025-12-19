using Test
include("../../src/OptAlgos.jl")
using .OptAlgos

"""
Wolfe Condition Line Search test
"""

# test for univariate function
@testset "Wolfe Line Search univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    c = ConvergenceCriteria()
    GradientEstimator = FiniteDifferenceUnivariate{5,:central}(1e-6)
    stepLengthStrategy = Backtracking(0.5)
    wolfeLineSearch = WolfeLineSearch(c1 = 1e-4, c2 = 0.999, stepLengthStrategy = stepLengthStrategy)
    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = lineSearch(f, x0, GradientEstimator, wolfeLineSearch, c, lim = 200, lineSearchLim = 500, printIter = true)
        @test isapprox(result.minimum, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "Wolfe Line Search multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    c = ConvergenceCriteria(gradTol=1e-4)
    GradientEstimator = FiniteDifferenceMultivariate{5,:central}(3e-6)
    stepLengthStrategy = Backtracking(0.9)
    stepLengthStrategy = QuadraticInterpolation()
    wolfeLineSearch = WolfeLineSearch(c1 = 1e-4, c2 = 0.9, stepLengthStrategy = stepLengthStrategy)
    for x0 in [[-100.0, 0.0], [-10.0, 20.0], [0.0, -100.0], [10.0, 10.0], [100.0, 50.0]]
        result = lineSearch(f, x0, GradientEstimator, wolfeLineSearch, c, alpha = 2, lim = 2000, lineSearchLim = 8000)
        @test isapprox(result.minimum, [0, 0], atol=1e-3) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-3) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = lineSearch(f3, [1.0, 1.0, 1.0], GradientEstimator, wolfeLineSearch, c) # test for minimum point x coordinate
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

