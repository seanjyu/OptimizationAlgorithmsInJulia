using Test
include("../../src/OptAlgos.jl")
using .OptAlgos

"""
Gradient Descent Tests
"""

# test for univariate function
@testset "Gradient Descent univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    GradientEstimator = FiniteDifferenceUnivariate{3,:forward}(1e-5)
    c = ConvergenceCriteria()
    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = GradientDescent(f, x0, GradientEstimator, c)
        @test isapprox(result.minimum, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "Gradient Descent multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    c = ConvergenceCriteria()
    GradientEstimator = FiniteDifferenceMultivariate{3,:central}(1e-5)
    for x0 in [[-10.0, 0.0], [0.0, -10.0], [10.0, 1.0], [10.0, 5.0]]
        result = GradientDescent(f, x0, GradientEstimator, c, lim = 700, tol = 1e-8) 
        
        @test isapprox(result.minimum, [0, 0], atol=1e-3) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-3) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = GradientDescent(f3, [1.0, 1.0, 1.0], GradientEstimator, c, track = false) 
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3)
end;

