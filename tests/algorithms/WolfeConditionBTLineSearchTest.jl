using Test
include("../../utils/GradientEstimators/GradientEstimatorType.jl")
include("../../utils/GradientEstimators/ForwardDifference.jl")
include("../../utils/TestFunctions.jl") 
using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
include("../../algorithms/lineSearch/lineSearchBase.jl")
include("../../algorithms/lineSearch/WolfeLineSearch.jl")
using .lineSearchInterface
using .lineSearchBaseModule
using .WolfeLineSearchModule

"""
Backtracking Armijo Line Search test
"""

# test for univariate function
@testset "Wolfe Line Search univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    GradientEstimator = UnivariateForwardDifference(1e-5)
    wolfeBTLineSearch = WolfeBTLineSearch(c1 = 1e-4, c2 = 0.999, rho = 0.5)
    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = lineSearch(f, x0, GradientEstimator, wolfeBTLineSearch, lim = 200, lineSearchLim = 500, printIter = true)
        @test isapprox(result.minimum, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.functionValues[end], 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "Wolfe Line Search multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    GradientEstimator = MultivariateForwardDifference(1e-5)
    wolfeBTLineSearch = WolfeBTLineSearch(c1 = 1e-4, c2 = 0.999, rho = 0.8)
    for x0 in [[-100.0, 0.0], [-10.0, 20.0], [0.0, -100.0], [10.0, 10.0], [100.0, 50.0]]
        result = lineSearch(f, x0, GradientEstimator, wolfeBTLineSearch, alpha = 2, lim = 500, lineSearchLim = 8000, tol =3e-5)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.functionValues[end], 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = lineSearch(f3, [1.0, 1.0, 1.0], GradientEstimator, wolfeBTLineSearch) # test for minimum point x coordinate
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

