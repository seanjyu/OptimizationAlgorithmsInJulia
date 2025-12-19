using Test
# include("../../utils/GradientEstimators/GradientEstimatorType.jl")
# include("../../utils/GradientEstimators/ForwardDifference.jl")
# include("../../utils/TestFunctions.jl") 
# using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
# using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
# include("../../algorithms/newton/NewtonMethod.jl")
# using .NewtonMethodModule
include("../../src/OptAlgos.jl")
using .OptAlgos

"""
Newton Method Tests
"""

# test for univariate function
@testset "Newton method univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    # GradientEstimator = UnivariateForwardDifference(1e-5)
    GradientEstimator = FiniteDifferenceUnivariate{5,:central}(1e-6)
    c = ConvergenceCriteria()
    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = NewtonMethod(f, x0, GradientEstimator, c)
        @test isapprox(result.minimum, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "Newton method multivariate objective function test" begin
    A = [2.0 3.0; 3.0 2.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    # GradientEstimator = MultivariateForwardDifference(1e-5)
    GradientEstimator = FiniteDifferenceMultivariate{5,:central}(3e-6)
    c = ConvergenceCriteria()
    for x0 in [[-100.0, 0.0], [-10.0, 20.0], [0.0, -100.0], [10.0, 10.0], [100.0, 50.0]]
        result = NewtonMethod(f, x0, GradientEstimator, c)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = NewtonMethod(f3, [1.0, 1.0, 1.0], GradientEstimator,  c)
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3)
end;

