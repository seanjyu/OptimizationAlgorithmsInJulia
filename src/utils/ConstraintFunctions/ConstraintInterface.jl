
    """
    Interface for constraint sets
    """
    abstract type Constraint end
    
    """
    project
        Projects a point onto the constraint set
    
    Input
        constraint::Constraint
        x (Vector) - point to project
        
    Output - closest feasible point
    """
    function project(constraint::Constraint, x)
        error("project is not implemented for $(typeof(constraint))")
    end
    
    """
    isFeasible
        Check if a point satisfies the constraint
    
    Input
        constraint::Constraint
        x (Vector) - point to check
        
    Output - Boolean
    """
    function isFeasible(constraint::Constraint, x)
        error("isFeasible is not implemented for $(typeof(constraint))")
    end

    """
    violation
        Calculate how much a point violates the constraint
    """
    function violation(constraint::Constraint, x)
        error("violation is not implemented for $(typeof(constraint))")
    end

    """
    residual
    """
    function residual(constraint::Constraint, x)
        error("residual is not implemented for $(typeof(constraint))")
    end

    """
    jacobian
        Calculate the jacobian of the constraint at a point. Note for multivariate constraints the jacobian will be returned
    """
    function jacobian(constraint::Constraint, x)
        error("jacobian is not implemented for $(typeof(constraint))")
    end

    """
    isInequality
        Simple function that returns a boolean value of whether the constraint is an inequality. Useful for barrier methods
    """
    function isInequality(constraint::Constraint)
        error("isInequality is not implemented for $(typeof(constraint))")
    end
    