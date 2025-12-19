struct MiniBatchGradientEstimator <: GradientEstimator
    data::AbstractMatrix       
    labels::AbstractVector     
    batchSize::Int
    gradMethod::GradientEstimator
end

function gradient(est::MiniBatchGradientEstimator, f, x)
    # # Sample mini-batch indices
    # N = size(est.data, 2)
    # indices = rand(1:N, est.batchSize)
    
    # # Create batch-specific loss: f(params, data, labels) -> f(params)
    # batchData = @view est.data[:, indices]
    # batchLabels = @view est.labels[indices]
    # # Replace with loss function
    # # fBatch(params) = f(params, batchData, batchLabels)
    
    # return gradient(est.gradMethod, fBatch, x)
    n = size(est.data, 2)
    idx = if est.batchSize >= n
        1:n  # Use all data
    else
        randperm(n)[1:est.batchSize]  # Sample without replacement
    end

    # idx = rand(1:size(est.data, 2), est.batchSize)


    X_batch = @view est.data[:, idx]
    y_batch = @view est.labels[idx]
    
    fBatch(p) = f(p, X_batch, y_batch)
    
    result = gradient(est.gradMethod, fBatch, x)
    return (grad = result.grad, funcEvals = result.funcEvals)
end

function hessian(est::MiniBatchGradientEstimator, f, x)
    # idx = rand(1:size(est.data, 2), est.batchSize)
    n = size(est.data, 2)
    idx = if est.batchSize >= n
        1:n  # Use all data
    else
        randperm(n)[1:est.batchSize]  # Sample without replacement
    end
    X_batch = @view est.data[:, idx]
    y_batch = @view est.labels[idx]
    
    fBatch(p) = f(p, X_batch, y_batch)
    
    result = hessian(est.gradMethod, fBatch, x)
    return (hess = result.hess, funcEvals = result.funcEvals, gradEvals = result.gradEvals)
end

