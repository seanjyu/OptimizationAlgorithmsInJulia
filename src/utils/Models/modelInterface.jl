abstract type Model end

# function predict(model::Model, params, X)
#     error("predict not implemented for $(typeof(model))")
# end

function initModel(model::Model)
    error("initModel not implemented for $(typeof(model))")
end