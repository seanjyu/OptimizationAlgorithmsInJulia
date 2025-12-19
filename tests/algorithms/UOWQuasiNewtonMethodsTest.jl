using Test
# include("../../utils/GradientEstimators/GradientEstimatorType.jl")
# include("../../utils/GradientEstimators/ForwardDifference.jl")
# include("../../utils/TestFunctions.jl")
# include("../../utils/UnconstrainedOptWrapper/UnconstrainedOptSolverInterface.jl") 
# using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
# using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
# using .UnconstrainedOptSolverInterface
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
# include("../../utils/UnconstrainedOptWrapper/QausiNewtonOptSolver.jl")
# using .QuasiNewtonOptSolver
include("../../src/OptAlgos.jl")
using .OptAlgos


"""
Unconstrained Optimization Wrapper Quasi-Newton Test
"""

# No need to test for univariate function

# test for multivariate function
@testset "SR1H with Wolfe condition line search multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    c = ConvergenceCriteria()
    
    GradientEstimator = FiniteDifferenceMultivariate{5,:central}(3e-6)
    stepLengthStrategy = Backtracking(0.8)
    wolfeLineSearch = WolfeLineSearch(c1 = 1e-4, c2 = 0.95, stepLengthStrategy = stepLengthStrategy, curvTol = 1e-16)
    SR1HParameters = SR1H()
    p = UOWQuasiNewtonOpt(GradientEstimator, SR1HParameters, wolfeLineSearch, c)
    
    for x0 in [[-1.0, 0.0], [-1.0, 2.0], [0.0, -1.0], [1.0, 1.0], [1.0, 5.0]]
        result = solveUnconstrainedOpt(f, x0, p)
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.finalValue, 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = solveUnconstrainedOpt(f3, [1.0, 1.0, 1.0], p)
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

