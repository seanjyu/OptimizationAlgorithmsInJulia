"""
Projected Gradient Descent

Reference(s)


Required Inputs     
    f (function) - Objective function 
    x0 (vector) - Starting coordinate 
    constraints (Constraint) - 
    gradEstimator (GradientEstimator) - Gradient estimator struct, see utils/GradientEstimatorInterface for more details    
    criteria (Criteria) - 

Optional Inputs
    track (boolean) - 
    alpha (Float64) - 
    lim (Int) - Maximum number of iterations

Output - named tuple with the following fields
    minimumPoint (Vector) - Final coordinate of algorithm 
    finalValue (Float64) - Objective function evaluation at final coordinate 
    logger (struct) - if track flag set to true then the following fields can be accessed
        path (Array) - Coordinates at each iteration 
        gradients (Array) - Gradient values at each iteration
        functionValues (Array) - Objective function values at each iteration
        iterations (Int) - Bumber of iterations 
        algorithmData (Struct) - Algorithm specific data

"""

function ProjectedGradientDescent(f, x0, gradEstimator::GradientEstimator, constraints::Constraint, criteria::ConvergenceCriteria; alpha=0.1, tol=1e-5, lim=100, track = true)
    x = x0
    fCur = f(x0)

    logger = initLogger(track, x0, fCur, lim, algorithmData = NoAlgorithmData())

    #TODO add algorithmData?
    
    for i in 1:lim

        gradRes = gradient(gradEstimator, f, x)
        grad = gradRes.grad

        xOld = copy(x)
        fOld = fCur
         
        # perform gradient descent step
        x = x .- alpha .* grad
        
        # perform projection
        x = project(constraints, x)

        fCur = f(x)  

        #TODO add stepsize to logIter
        # logIter!(logger, fCur, x, grad, x,  1 + gradRes.funcEvals, gradRes.funcEvals, 1)

        converged, reason = CheckConvergence(criteria, grad, x, xOld, fCur, fOld, i)
        if converged 
            setConvergenceReason!(logger, reason)
            break
        end

    end
    finalizeLogger!(logger)
    return (
            minimumPoint = x,
            finalValue = fCur,
            logger = logger
        )
end
