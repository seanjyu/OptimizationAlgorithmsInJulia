using Test
using Statistics
include("../../../src/OptAlgos.jl")
using .OptAlgos


@testset "MiniBatchGradientEstimator Tests" begin
    
    # Synthetic linear regression data
    n_features = 3
    n_samples = 100
    X = randn(n_features, n_samples)
    true_weights = [1.0, -2.0, 0.5]
    y = vec(true_weights' * X) + 0.1 * randn(n_samples)
    
    model = LinearRegressionModel(n_features)
    loss = modelLossFunction(model, MSE())

    
    # Use your existing finite difference estimator
    fd_estimator = FiniteDifferenceMultivariate{3,:central}(1e-5)
    minibatch_est = MiniBatchGradientEstimator(X, y, 16, fd_estimator)
    
    
    x0 = randn(n_features)
    
    @testset "Gradient structure" begin
        result = gradient(minibatch_est, loss, x0)
        
        @test haskey(result, :grad)
        @test haskey(result, :funcEvals)
        @test length(result.grad) == n_features
        @test result.funcEvals == 2 * n_features  # 3-point central uses 2n evals
    end
    
    @testset "Hessian structure" begin
        result = hessian(minibatch_est, loss, x0)
        
        @test haskey(result, :hess)
        @test size(result.hess) == (n_features, n_features)
        @test result.hess ≈ result.hess' atol=1e-6  # Symmetry
    end
    
    @testset "Stochastic behavior" begin
        grads = [gradient(minibatch_est, loss, x0).grad for _ in 1:10]
        # Different batches should yield different gradients
        @test !all(g ≈ grads[1] for g in grads[2:end])
    end
    
    @testset "Full batch matches direct computation" begin
        full_batch_est = MiniBatchGradientEstimator(X, y, n_samples, fd_estimator)
        
        full_result = gradient(full_batch_est, loss, x0)
        direct_result = gradient(fd_estimator, p -> loss(p, X, y), x0)
        
        @test full_result.grad ≈ direct_result.grad atol=1e-10
    end
    
    @testset "Gradient variance decreases with batch size" begin
        function gradient_variance(batch_size, n_trials=50)
            est = MiniBatchGradientEstimator(X, y, batch_size, fd_estimator)
            grads = [gradient(est, loss, x0).grad for _ in 1:n_trials]
            return mean(var(g[i] for g in grads) for i in 1:n_features)
        end
        
        var_small = gradient_variance(8)
        var_large = gradient_variance(64)
        
        @test var_large < var_small
    end
end