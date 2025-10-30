module UnivariateQuadraticFunctionModule

    export UnivariateQuadraticFunction, discriminant, roots, trueGradient

    struct UnivariateQuadraticFunction
        a::Real
        b::Real
        c::Real
    end

    function (qf::UnivariateQuadraticFunction)(x)
        return qf.a * x.^2 + qf.b * x + qf.c
    end

    function discriminant(qf::UnivariateQuadraticFunction)
        return qf.b^2 - 4*qf.a*qf.c
    end

    function roots(qf::UnivariateQuadraticFunction)
        disc = discriminant(qf)
        if disc < 0
            return "No real roots"
        else
            sqrt_disc = sqrt(disc)
            root1 = (-qf.b + sqrt_disc) / (2*qf.a)
            root2 = (-qf.b - sqrt_disc) / (2*qf.a)
            return (root1, root2)
        end
    end

    function trueGradient(qf::UnivariateQuadraticFunction, x)
        return [2 .* qf.a .* x + qf.b]
    end

end  

module MultivariateQuadraticFunctionModule

    using LinearAlgebra

    export MultivariateQuadraticFunction, trueGradient, eigenvalues, trueMinimum
    
    struct MultivariateQuadraticFunction{T<:Number}
        A::AbstractMatrix{T}
        b::AbstractVector{T}
        c::T

        # Implement several constructors depending on input values
        # A, B, C in input
        function MultivariateQuadraticFunction(A::AbstractMatrix{T}, b::AbstractVector{T}, c::T) where T
            isapprox(A, A'; atol=1e-8) || throw(ArgumentError("Input matrix must be symmetric"))
            size(A, 1) == size(A, 2) || throw(DimensionMismatch("Input A must be square"))
            length(b) == size(A, 1) || throw(DimensionMismatch("Input b must match A dimensions"))
            new{T}(A, b, c)
        end
        
        # Only A
        function MultivariateQuadraticFunction(A::AbstractMatrix{T}) where T
            b = zeros(T, size(A, 1))
            c = zero(T)
            MultivariateQuadraticFunction(A, b, c)
        end
        
        # Only A and b
        function MultivariateQuadraticFunction(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
            c = zero(T)
            MultivariateQuadraticFunction(A, b, c)
        end

        # Only A and c
        function MultivariateQuadraticFunction(A::AbstractMatrix{T}, c::T) where T
            b = zeros(T, size(A, 1))
            MultivariateQuadraticFunction(A, b, c)
        end
    end

    function (qf::MultivariateQuadraticFunction)(x::AbstractVector{T}) where T
        return 0.5 * x' * qf.A * x + qf.b' * x + qf.c 
    end

    function trueGradient(qf::MultivariateQuadraticFunction, x::AbstractVector{T}) where T
        return qf.A * x + qf.b
    end

    function eigenvalues(qf::MultivariateQuadraticFunction)
        return eigen(Symmetric(qf.A)).values
    end

    function trueMinimum(qf::MultivariateQuadraticFunction)
        return - (qf.A \ qf.b)  # Solve: Ax + b = 0
    end

end



function testF(x)
	a = 2*x + 2*x^2
	a
end	