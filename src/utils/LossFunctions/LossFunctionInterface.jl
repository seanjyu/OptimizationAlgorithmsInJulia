abstract type LossFunction end

function computeLoss(lossFunction::LossFunction, f, x)
    error("computeLoss not implemented for $(typeof(LossFunction))")
end

```
Helper function 
```
function modelLossFunction(model::Model, loss::LossFunction)
    return (params, X, y) -> computeLoss(loss, x -> model(params, x), X, y)
end