struct LogBarrier <: BarrierMethod end

function barrierValue(::LogBarrier, violation)
    any(v -> v >= 0, violation) && return Inf
    return -sum(v -> log(-v), violation)
end