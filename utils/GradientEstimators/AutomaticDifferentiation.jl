module UnivariateForwardAutoDiffModule

    export D

    # Define a dual number-like type D for forward-mode AD
    struct D <: Number
        f::Tuple{Float64, Float64}  # (value, derivative)
    end

    # Constructor for D
    D(x::Float64, y::Float64) = D((x, y))

    # Import Base methods to overload
    import Base: +, -, *, /, ^, >, convert, promote_rule

    # Addition
    +(x::D, y::D) = D(x.f[1] + y.f[1], x.f[2] + y.f[2])

    # Subtraction
    -(x::D, y::D) = D(x.f[1] - y.f[1], x.f[2] - y.f[2])

    # Multiplication
    *(x::D, y::D) = D(x.f[1] * y.f[1], x.f[2] * y.f[1] + x.f[1] * y.f[2])

    # Division
    /(x::D, y::D) = D(x.f[1] / y.f[1], (x.f[2] * y.f[1] - x.f[1] * y.f[2]) / y.f[1]^2)

    # Greater-than (for comparisons)
    >(x::D, y::D) = x.f[1] > y.f[1]

    # Conversion from real numbers (for mixed ops like D + 1.0)
    convert(::Type{D}, x::Real) = D(x, 0.0)

    # Promote rule so D and Number work together in expressions
    promote_rule(::Type{D}, ::Type{<:Number}) = D

end


#TODO
# module MultviariateForwardAutoDiffModule

#     struct D <: Number  # D is a function-derivative pair
#         f::Tuple{Float64,Float64}
#     end

#     D(x,y) = D((x,y))

# 	import Base: +, -, *, /, ^, >, convert, promote_rule
# 	+(x::D, y::D) = D(x.f .+ y.f)
# 	/(x::D, y::D) = D((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
# 	-(x::D, y:: D) = D(x.f .- y.f)
# 	*(x::D, y::D) = D((x.f[1]*y.f[1], (x.f[2]*y.f[1] + x.f[1]*y.f[2])))

#     >(x::D, y::D) = x.f[1] > y.f[1]

#     convert(::aType{D}, x::Real) = D((x,zero(x)))
# 	promote_rule(::Type{D}, ::Type{<:Number}) = D

# end

