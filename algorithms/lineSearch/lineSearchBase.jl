module lineSearchInterface
    using ..GradientEstimatorInterface: GradientEstimator
    export LineSearchMethod, stepSearch

    """
    Interface Module for Line Search methods
        Type: 
            LineSearchMethod - In concrete implementation assign type to struct and store hyperparameters
            
        Function(s):
            stepSearch - Method interface to perform step search based on specific line search method
    """
    abstract type LineSearchMethod end

    """
    stepSearch
        Method interface for generic line search method.
        Note not all concrete implementations require all inputs but they have been included for generalization purposes.
    Input
        lineSearchMethod (LineSearchMethod) - Struct containing hyperparameters for the specific line search method 
        gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details 
        f (function) - Objective function to be optimized 
        direction (Vector) - current step direction 
        xCur (Vector) - Current coordinate
        fCur (Number) - Objective function evaluation at current coordinate 
        gradCur (Vector) - Current gradient 
        alpha (Float64) - Initial step length 
        lineSearchLim (Int) - Iteration limit for line search 

    Output - named tuple with the following fields
        xNew (Vector) - New coordinate 
        fFinal (Vector) - Objective function evaluation at new coordinate 
        alphaFinal (Float64) - Final step size 
    """
    function stepSearch(lineSearchMethod::LineSearchMethod, 
                        gradEstimator::GradientEstimator, 
                        f, 
                        direction, 
                        xCur, 
                        fCur,
                        gradCur, 
                        alpha,  
                        lineSearchLim)
        error("step search not implemented for $(typeof(lineSearchMethod))")
    end
end

module lineSearchBaseModule

    using ..GradientEstimatorInterface: GradientEstimator, gradient
    using LinearAlgebra 
    using ..lineSearchInterface:LineSearchMethod, stepSearch
    
    export lineSearch

    function lineSearch(f, x0, gradEstimator::GradientEstimator, lineSearchMethod::LineSearchMethod; alpha = 1, tol = 1e-4, lim = 100, lineSearchLim = 100, printIter = false)
        """
        lineSearch
            Implementation of generic line search

        Input:
            f (function) - Objective function 
            x0 (vector) - Starting coordinate 
            gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details
            lineSearchMethod (LineSearchMethod)- LineSearchMethod struct impementing lineSearchInterface with a specific step length algorithm 
            alpha (Float64) - Initial step length
            tol (Float64) - Stop criteria, if norm grad smaller than tolerance value iterations will stop
            lim (Int) - Maximum number of iterations

        Output - named tuple with the following fields
            minimum (Vector)- Final coordinate 
            path (Array) - Coordinates at each iteration 
            gradients (Array) - Gradient values at each iteration
            functionValues (Array) - Objective function values at each iteration
            iterations (Int) - Bumber of iterations 
        """
        x = x0
        path = [copy(x0)]
        gradients = []
        functionValues = [f(x0)]
        stepSizes = []

        for i in 1:lim
            grad = gradient(gradEstimator, f, x)
            

            if norm(grad) < tol
                if printIter
                    println("Line search converged in $i iterations. Final gradient norm: $(norm(grad))")
                end
                return (
                    minimum = x,
                    path = path,
                    gradients = gradients,
                    functionValues = functionValues,
                    iterations = i,
                    stepSizes = stepSizes
                )
            end

            push!(gradients, copy(grad))
            
            fCur = functionValues[end]

            stepSearchResult = stepSearch(lineSearchMethod, gradEstimator, f, -grad, x, fCur, grad, alpha, lineSearchLim)

            x = stepSearchResult.xNew
            
            push!(path, copy(x))
            push!(functionValues, stepSearchResult.fFinal)
            push!(stepSizes, stepSearchResult.alphaFinal)

        end
        error("line search failed: maximum iterations ($lim) reached")
    end
end

