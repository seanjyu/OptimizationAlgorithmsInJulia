struct LinearRegressionModel <: Model
    inputDim::Int
end

function initModel(m::LinearRegressionModel)
    return zeros(m.inputDim)
end

```
Prediction function - functor such that can just 
```
(m::LinearRegressionModel)(params, X) = X' * params