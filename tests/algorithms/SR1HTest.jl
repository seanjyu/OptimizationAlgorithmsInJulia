using Test
# include("../../utils/GradientEstimators/GradientEstimatorType.jl")
# include("../../utils/GradientEstimators/ForwardDifference.jl")
# include("../../utils/TestFunctions.jl") 
# using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
# using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
# include("../../algorithms/lineSearch/lineSearchBase.jl")
# include("../../algorithms/lineSearch/WolfeLineSearch.jl")
# include("../../algorithms/newton/QuasiNewtonBase.jl")
# include("../../algorithms/newton/SR1.jl")
# using .lineSearchInterface
# using .lineSearchBaseModule
# using .WolfeLineSearchModule
# using .QuasiNewtonInterface
# using .QuasiNewtonBaseModule
# using .SR1Module
include("../../src/OptAlgos.jl")
using .OptAlgos
"""
SR1H Search test
"""

# No need to test for univariate function

# test for multivariate function
@testset "SR1H with Wolfe condition line search multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    c = ConvergenceCriteria(gradTol = 1e-6)
    GradientEstimator = FiniteDifferenceMultivariate{5,:central}(1e-6)
    stepLengthStrategy = QuadraticInterpolation()
    wolfeLineSearch = WolfeLineSearch(c1 = 1e-4, c2 = 0.9, stepLengthStrategy = stepLengthStrategy)
    SR1HParameters = SR1H()

    for x0 in [[-1.0, 0.0], [-1.0, 2.0], [0.0, -1.0], [1.0, 1.0], [1.0, 5.0]]
        result = QuasiNewtonOpt(f, x0, GradientEstimator, SR1HParameters, wolfeLineSearch, c, alpha = 1, lim = 500, lineSearchLim =1000, tol = 1e-5)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = QuasiNewtonOpt(f3, [1.0, 1.0, 1.0], GradientEstimator, SR1HParameters, wolfeLineSearch, c)
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

