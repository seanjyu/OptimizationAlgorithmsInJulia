using Test
# include("../../utils/GradientEstimators/GradientEstimatorType.jl")
# include("../../utils/GradientEstimators/ForwardDifference.jl")
# include("../../utils/TestFunctions.jl") 
# using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
# using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
# include("../../algorithms/lineSearch/lineSearchBase.jl")
# include("../../algorithms/lineSearch/WolfeLineSearch.jl")
# include("../../algorithms/ConjugateGradient/NonlinearCGBase.jl")
# include("../../algorithms/ConjugateGradient/HestenesStiefel.jl")
# using .lineSearchInterface
# using .lineSearchBaseModule
# using .WolfeLineSearchModule
# using .NonlinearCGInterface
# using .NonlinearCGBaseModule
# using .HestenesStiefelModule
# using LinearAlgebra


# include("../../utils/GradientEstimators/JuliaAutoDiff.jl")
# using .JuliaAutoDiffModule


"""
Fletcher Reeves test
"""

# No need to test for univariate function

# test for multivariate function
@testset "Fletcher Reeves multivariate objective function test" begin
    A = [2.0 3.0; 3.0 5.0]
    f = MultivariateQuadraticFunction(A, 2.0)
    # GradientEstimator = MultivariateForwardDifference(1e-6)
    
    GradientEstimator = ForwardDiffEstimator()
    wolfeBTLineSearch = WolfeBTLineSearch(c1 = 1e-4, c2 = 0.8, rho = 0.95, curvTol = 1e-4, strong = true)
    hestenesStiefel = HestenesStiefel()
    println("cond number: $(cond(A))")
    for x0 in [[-1.0, 0.0], [-1.0, 2.0], [0.0, -1.0], [1.0, 1.0], [1.0, 5.0]]
        result = NonlinearCGMethod(f, x0, GradientEstimator, hestenesStiefel, wolfeBTLineSearch, alpha = 0.8, lim = 1000, lineSearchLim =1000, tol = 1e-4, printIter = true)
        # println(result.path[20])
        # println(result.gradients[10])
        # println(result.functionValues[10])
        @test isapprox(result.minimum, [0, 0], atol=1e-4) # test for minimum point x coordinate
        @test isapprox(result.functionValues[end], 2.0, atol=1e-4) # test for minimum point y coordinate
    end

    # # 3D
    A3 = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
    f3 = MultivariateQuadraticFunction(A3)
    result3 = NonlinearCGMethod(f3, [1.0, 1.0, 1.0], GradientEstimator, hestenesStiefel, wolfeBTLineSearch,alpha = 0.5, lim = 1000, lineSearchLim = 5000, tol = 1e-5, printIter = true)
    @test isapprox(result3.minimum, [0.0, 0.0, 0.0], atol=1e-3) # test for minimum point y coordinate
end;

