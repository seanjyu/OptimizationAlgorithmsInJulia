using Test
include("../../src/OptAlgos.jl")
using .OptAlgos

"""
Newton Method Tests
"""

# test for univariate function
@testset "Newton method univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    GradientEstimator = UnivariateForwardDifference(1e-5) 
    c = ConvergenceCriteria()
    p = NewtonOpt(GradientEstimator, c)
    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = solveUnconstrainedOpt(f, x0, p)
        @test isapprox(result.minimum, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "Newton method multivariate objective function test" begin
    A = [2.0 3.0; 3.0 2.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    GradientEstimator = MultivariateForwardDifference(1e-5)
    c = ConvergenceCriteria()
    p = NewtonOpt(GradientEstimator, c)
    for x0 in [[-100.0, 0.0], [-10.0, 20.0], [0.0, -100.0], [10.0, 10.0], [100.0, 50.0]]
        result = solveUnconstrainedOpt(f, x0, p)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    x03D = [1.0, 1.0, 1.0]
    result3 = solveUnconstrainedOpt(f3, x03D , p)
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3)
end;

