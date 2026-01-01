using Test
include("../../src/OptAlgos.jl")
using .OptAlgos

"""
More Thuente Line Search test
"""

# test for univariate function
@testset "Armijo Line Search univariate objective function test" begin
    f = UnivariateQuadraticFunction(1, 1, 100)
    c = ConvergenceCriteria()
    GradientEstimator = FiniteDifferenceUnivariate{5,:central}(1e-6)
    moreThuenteLineSearch = MoreThuenteLineSearch()

    for x0 in [-100.0, -10.0, 0.0, 10.0, 100.0]
        result = lineSearch(f, x0, GradientEstimator, moreThuenteLineSearch, c, lim = 200, lineSearchLim = 500, printIter = true)
        @test isapprox(result.minimumPoint, -0.5, atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 99.75, atol=1e-4) # test for minimum point y coordinate
    end
end;


# test for multivariate function
@testset "Armijo Line Search multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    c = ConvergenceCriteria()
    GradientEstimator = FiniteDifferenceMultivariate{5,:central}(3e-6)
    moreThuenteLineSearch = MoreThuenteLineSearch()
    for x0 in [[-10.0, 0.0], [-10.0, 2.0], [0.0, -10.0], [10.0, 1.0], [10.0, 5.0]]
        result = lineSearch(f, x0, GradientEstimator, moreThuenteLineSearch, c,  alpha = 2, lim = 500, lineSearchLim = 200)
        @test isapprox(result.minimumPoint, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = lineSearch(f3, [1.0, 1.0, 1.0], GradientEstimator, moreThuenteLineSearch, c) # test for minimum point x coordinate
    @test isapprox(result3.minimumPoint, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

