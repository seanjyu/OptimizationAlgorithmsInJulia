struct ConvergenceCriteria
    gradTol::Float64
    xTol::Float64
    fTol::Float64
    stepTol::Float64
    all::Bool
    reason::Bool
end

# By default only check for gradient convergence
ConvergenceCriteria(; 
    gradTol=1e-6,
    xTol=0.0,
    fTol=0.0,
    stepTol=0.0,
    all=false,
    reason = true,
) = ConvergenceCriteria(gradTol, xTol, fTol, stepTol, all, reason)

function CheckConvergence(c::ConvergenceCriteria, grad, xCur, xOld, fCur, fOld, step)
    if c.all
        # calculate all that have defined tolerance
        gNorm = (c.gradTol != 0.0) ? norm(grad) : 0.0
        xDiff = (c.xTol != 0.0) ? norm(xCur - xOld) : 0.0
        fDiff = (c.fTol != 0.0) ? abs(fCur - fOld) : 0.0
        stepSize = (c.stepTol != 0.0) ? norm(step) : 0.0

        allConverged = (c.gradTol == 0.0 || gNorm < c.gradTol) &&
               (c.xTol == 0.0 || xDiff < c.xTol) &&
               (c.fTol == 0.0 || fDiff < c.fTol) &&
               (c.stepTol == 0.0 || stepSize < c.stepTol)

        if allConverged
              return true, msg
        else
            return false, ""
        end

    else
        if c.gradTol > 0.0 && (gNorm = norm(grad); gNorm < c.gradTol)
            msg = c.reason ? "Grad converged with: $(@sprintf("%.3e", gNorm))" : ""
            return true, msg
        end

        if c.xTol > 0.0 && (xDiff = norm(xCur - xOld); xDiff < c.xTol)
            msg = c.reason ? "x converged with: $(@sprintf("%.3e", xDiff))" : ""
            return true, msg
        end
        
        if c.fTol > 0.0 && (fDiff = abs(fCur - fOld); fDiff < c.fTol)
            msg = c.reason ? "function value converged with: $(@sprintf("%.3e", fDiff))" : ""
            return true, msg
        end

        if c.stepTol > 0.0 && (stepSize = norm(step); stepSize < c.stepTol)
            msg = c.reason ? "stepSize converged with: $(@sprintf("%.3e", stepSize))" : ""
            return true, msg
        end
        
        return false, ""
    end
end
