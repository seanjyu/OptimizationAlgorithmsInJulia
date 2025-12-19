module ConjugateGradientModule
    using LinearAlgebra
    export ConjugateGradient, PreconditionConjugateGradient

    """
    Conjugate Gradient Method
        Implementation of basic Conjugate Gradient method

    """
    function ConjugateGradient(A::Matrix, b::Vector, x0::Vector; tol = 1e-6, lim = 100, track = true)
        x = copy(x0)
        r = b - A * x 
        p = r
        # k = 0
        # for traditional conjugate gradient the convergence criteria is when the norm of the residual between steps has converged,  
        c = ConvergenceCriteria(gradTol = tol)
        converged = false
        reason = ""

        # logger tracks residuals
        algorithmData = track ? 
        AlgorithmData(lim; residuals=Vector{Float64}) : 
        NoAlgorithmData()
    
        logger = initLogger(track, x0, 0, lim; algorithmData=algorithmData)
        
        # while norm(r) > tol
        # while !converged
        for i in 1:lim

            rtr = r' * r
            a = rtr / (p' * A * p)
            x = x + a * p
            r = r - a * A * p
            beta = (r' * r) / rtr
            p = r + beta * p
            # k += 1
            
            # In the basic conjugate gradient no function evaluations or gradient 
            logIter!(logger, 0, x, 0, 0, 0, residuals=r)

            # check for convergence
            converged, reason = CheckConvergence(c, norm(r), 0, 0, 0, 0, 0)
            if converged
                setConvergenceReason!(logger, reason)
                finalizeLogger!(logger)

                return (minimum = x, logger = logger)
            end
            
        end

    end

    """
    Preconditioned Conjugate Gradient Method
        Implementation of Preconditioned Conjugate Gradient method 
    """
    function PreconditionConjugateGradient(A::Matrix, b::Vector, x0::Vector, M::Matrix; tol = 1e-6)
        x = copy(x0)
        r = b - A * x
        y = M \ r        
        p = y
        k = 0

        while norm(r) > tol
            rty = r' * y
            a = rty / (p' * A * p)
            x = x + a * p
            r = r - a * A * p
            y = M \ r
            beta = (r' * y) / rty
            p = y + beta * p
            k += 1
        end

        return x, k
    end

end