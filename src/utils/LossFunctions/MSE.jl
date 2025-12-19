struct MSE <: LossFunction
end

function computeLoss(lossFunction::MSE, f, X, y)
    prediction = f(X)
    return sum((prediction .- y).^2) / length(y)
end