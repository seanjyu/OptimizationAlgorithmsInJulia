"""
ForwardDiffEstimator

Uses ForwardDiff.jl for automatic differentiation.

# Fields
- `chunk_size::Union{Int,Nothing}`: Chunk size for ForwardDiff (nothing = automatic)
- `cache_config::Bool`: Whether to cache ForwardDiff configuration
"""
struct ForwardDiffEstimator <: GradientEstimator
    chunk_size::Union{Int,Nothing}
    cache_config::Bool
    
    ForwardDiffEstimator(; chunk_size = nothing, cache_config = false) = 
        new(chunk_size, cache_config)
end

"""
ReverseDiffEstimator

Uses ReverseDiff.jl for automatic differentiation.
Best for: functions where output_dim >> input_dim (rare in optimization).
Only supports gradients efficiently, not Hessians.

# Fields
- `compile::Bool`: Whether to compile the tape for repeated evaluations
"""
struct ReverseDiffEstimator <: GradientEstimator
    compile::Bool
    
    ReverseDiffEstimator(; compile = false) = new(compile)
end


"""
Forward Difference Gradient function
"""
function gradient(est::ForwardDiffEstimator, f, x::Vector{<:Real})
    if est.cache_config
        # Create gradient configuration for efficient repeated calls
        cfg = ForwardDiff.GradientConfig(f, x, 
            isnothing(est.chunk_size) ? ForwardDiff.Chunk(x) : ForwardDiff.Chunk(est.chunk_size))
        return ForwardDiff.gradient(f, x, cfg)
    else
        # Simple call without caching
        if isnothing(est.chunk_size)
            return ForwardDiff.gradient(f, x)
        else
            cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk(est.chunk_size))
            return ForwardDiff.gradient(f, x, cfg)
        end
    end
end

"""

"""
function hessian(est::ForwardDiffEstimator, f, x::Vector{<:Real})
    if est.cache_config
        cfg = ForwardDiff.HessianConfig(f, x,
            isnothing(est.chunk_size) ? ForwardDiff.Chunk(x) : ForwardDiff.Chunk(est.chunk_size))
        return ForwardDiff.hessian(f, x, cfg)
    else
        if isnothing(est.chunk_size)
            return ForwardDiff.hessian(f, x)
        else
            cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk(est.chunk_size))
            return ForwardDiff.hessian(f, x, cfg)
        end
    end
end


function gradient(est::ReverseDiffEstimator, f, x::Vector{<:Real})
    if est.compile
        # Compile tape for faster repeated evaluations
        tape = ReverseDiff.GradientTape(f, x)
        compiled_tape = ReverseDiff.compile(tape)
        result = similar(x)
        ReverseDiff.gradient!(result, compiled_tape, x)
        return result
    else
        # Simple call without compilation
        return ReverseDiff.gradient(f, x)
    end
end

function hessian(est::ReverseDiffEstimator, f, x::Vector{<:Real})
    # ReverseDiff doesn't have efficient Hessian computation
    # Fall back to ForwardDiff for Hessians
    @warn "ReverseDiff doesn't efficiently compute Hessians, using ForwardDiff instead"
    return ForwardDiff.hessian(f, x)
end

