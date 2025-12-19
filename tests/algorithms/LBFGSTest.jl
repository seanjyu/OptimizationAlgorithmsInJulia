using Test
include("../../src/OptAlgos.jl")
using .OptAlgos
"""
LBFGS Search test
"""

# No need to test for univariate function

# test for multivariate function
@testset "LBFGS with Wolfe condition line search multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    c = ConvergenceCriteria(gradTol = 1e-6)
    GradientEstimator = FiniteDifferenceMultivariate{5,:central}(1e-6)
    stepLengthStrategy = QuadraticInterpolation()
    wolfeLineSearch = WolfeLineSearch(c1 = 1e-4, c2 = 0.95, stepLengthStrategy = stepLengthStrategy, curvTol = 1e-16)
    
    for x0 in [[-1.0, 0.0], [-1.0, 2.0], [0.0, -1.0], [1.0, 1.0], [1.0, 5.0]]
        result = LBFGS(f, x0, GradientEstimator, wolfeLineSearch, c, m = 20, alpha = 3, lim = 500, lineSearchLim =70, tol = 1e-4)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = LBFGS(f3, [1.0, 1.0, 1.0], GradientEstimator, wolfeLineSearch, c)
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

