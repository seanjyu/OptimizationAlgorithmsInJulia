

"""
Logger for optimization algorithms
"""

# Abstract type for algorithm-specific data
abstract type AbstractAlgorithmData end

struct NoAlgorithmData <: AbstractAlgorithmData end


# """
# Note can store multiple vectors of tuples
# e.g.
# algorithmData = AlgorithmData(lim; 
#     data1 = Tuple{Float64, Float64},
#     data2 = Float64    
# )

# then during Logging
# logIter!(logger, fCur, x, grad, stepLength, fEvals, gradEstFuncEvals, gradEvals; 
#         data1 = (0.1, 0.2)
#         data2 = 0.1
#         )
# """
struct AlgorithmData{T} <: AbstractAlgorithmData
    data::T  # T will be a NamedTuple of Vectors
end

# Constructor for AlgorithmData
# function AlgorithmData(maxIter::Int; kwargs...)
#     data = NamedTuple{keys(kwargs)}(
#         [Vector{valtype(v)}(undef, maxIter) for v in values(kwargs)]   
#     )
#     return AlgorithmData(data)
# end

function AlgorithmData(maxIter::Int; kwargs...)
    data = (; (k => Vector{T}(undef, maxIter) for (k, T) in kwargs)...)
    return AlgorithmData(data)
end

struct NoLogger end

# Logger struct
mutable struct Logger{T, X, A <: AbstractAlgorithmData}
    path::Vector{X}
    gradients::Vector{X}
    functionValues::Vector{T}
    stepLengths::Vector{T}  

    # counters - Ref used for efficiency
    iterations::Ref{Int}
    functionEvals::Ref{Int}
    gradEstFunctionEvals::Ref{Int}
    gradientEvals::Ref{Int}
    
    convergenceReason::Ref{String}  # Must be Ref for setConvergenceReason! to work
    algorithmData::A
end

# Initialize NoLogger
function initLogger(::Type{NoLogger}, args...; kwargs...)
    return NoLogger()
end

# Initialize Logger
function initLogger(::Type{Logger}, x0, f0, maxIter::Int; 
                   algorithmData::A=NoAlgorithmData()) where {A <: AbstractAlgorithmData}

    T = eltype(x0) 
    X = typeof(x0)
    
    path = Vector{X}(undef, maxIter + 1)
    gradients = Vector{X}(undef, maxIter)
    functionValues = Vector{T}(undef, maxIter + 1)
    stepLengths = Vector{T}(undef, maxIter)
     
    path[1] = copy(x0)
    functionValues[1] = f0

    return Logger{T, X, A}(
        path, gradients, functionValues, stepLengths, 
        Ref(0),  # iterations
        Ref(0),  # function evals total (with grad estimator)
        Ref(0),  # function evals when calculating grad 
        Ref(0),  # grad evals
        Ref(""),  # convergence reason 
        algorithmData
    )
end

# Convenience constructor
function initLogger(track::Bool, x0, f, maxIter::Int; 
                   algorithmData::A=NoAlgorithmData()) where {A <: AbstractAlgorithmData}
    return track ? initLogger(Logger, x0, f, maxIter; algorithmData=algorithmData) : 
                   initLogger(NoLogger)
end

# Log iteration - NoLogger
logIter!(::NoLogger, args...; kwargs...) = nothing

# Log iteration - Logger
function logIter!(logger::Logger, 
                fCur, 
                x, 
                grad, 
                stepLength, 
                funcEvals, 
                gradEstFuncEvals, 
                gradEvals; 
                kwargs...)
    logger.iterations[] += 1
    i = logger.iterations[]
    logger.path[i + 1] = copy(x)
    logger.gradients[i] = copy(grad)
    logger.functionValues[i + 1] = fCur
    logger.stepLengths[i] = copy(stepLength)
    logger.functionEvals[] += funcEvals
    logger.gradEstFunctionEvals[] += gradEstFuncEvals
    logger.gradientEvals[] += gradEvals    

    logAlgorithmSpecific!(logger.algorithmData, i; kwargs...)  
    
    return nothing
end

# Finalize - NoLogger
finalizeLogger!(::NoLogger) = nothing

function finalizeLogger!(logger::Logger)  
    i = logger.iterations[]
    resize!(logger.path, i + 1)
    resize!(logger.gradients, i)
    resize!(logger.functionValues, i + 1) 
    resize!(logger.stepLengths, i)
    resizeAlgorithmData!(logger.algorithmData, i)  # Fixed name

    return logger
end

logAlgorithmSpecific!(::NoAlgorithmData, i; kwargs...) = nothing

function logAlgorithmSpecific!(data::AlgorithmData, i; kwargs...)
    for (key, value) in pairs(kwargs)
        if haskey(data.data, key)
            data.data[key][i] = value
        end
    end
end

# resize method for finalization
resizeAlgorithmData!(::NoAlgorithmData, i) = nothing

function resizeAlgorithmData!(data::AlgorithmData, i)
    for vec in values(data.data)
        resize!(vec, i)
    end
end

# Set convergence reason
setConvergenceReason!(::NoLogger, reason::String) = nothing

function setConvergenceReason!(logger::Logger, reason::String)
    logger.convergenceReason[] = reason 
end