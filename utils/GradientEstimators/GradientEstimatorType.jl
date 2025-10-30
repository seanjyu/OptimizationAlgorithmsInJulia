module GradientEstimatorInterface 
    export GradientEstimator, gradient, hessian
    """
        GradientEstimator
    Abstract type to store necessary information to produce gradient esitmate, e.g. delta for finite difference
    """
    abstract type GradientEstimator end

    """
        gradient(estimator::GradientEstimator, f, x)

    Compute the gradient of function `f` at point `x` using the specified estimator.
    All concrete subtypes must implement this method.
    """
    function gradient(estimator::GradientEstimator, f, x)
        throw(MethodError("gradient not implemented for $(typeof(estimator))"))
    end

    # function gradient()


    """
        hessian(estimator::GradientEstimator, f, x)

    Compute the hessian of function `f`
    """
    function hessian(estimator::GradientEstimator, f, x)
        throw(MethodError("hessian not implemented for $(typeof(estimator))"))
    end

end

# module LossGradientEstimatorInterface
#     export LossGradientEstimator, gradient, hessian
    
#     abstract type LossGradientEstimator end

#     function gradient(estimator::LossGradientEstimator, lossFunc, theta, data)
#         throw(MethodError("gradient not implemented for $(typeof(estimator))"))
#     end

#     function hessian(estimator::LossGradientEstimator, lossFunc, theta, data)
#         throw(MethodError("hessian not implemented for $(typeof(estimator))"))
#     end    
# end
