using Test
include("../../utils/GradientEstimators/GradientEstimatorType.jl")
include("../../utils/GradientEstimators/ForwardDifference.jl")
include("../../utils/TestFunctions.jl") 
using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
# IMPORT OPTIMIZATION ALGORITHM 

"""
Template testing file
"""

# test for univariate function
@testset "METHOD univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    GradientEstimator = UnivariateForwardDifference(1e-5)
    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = METHODNAME(f, x0, GradientEstimator)
        @test isapprox(result.minimum, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.functionValues[end], 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "METHOD multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    GradientEstimator = MultivariateForwardDifference(1e-5)
    for x0 in [[-100.0, 0.0], [-10.0, 20.0], [0.0, -100.0], [10.0, 10.0], [100.0, 50.0]]
        result = METHODNAME(f, x0, GradientEstimator)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.functionValues[end], 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # test 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3D = MultivariateQuadraticFunction(A3)
    x03d = [1.0, 1.0, 1.0]
    result3 = METHODNAME(f3D, x03d , GradientEstimator) # test for minimum point x coordinate
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

