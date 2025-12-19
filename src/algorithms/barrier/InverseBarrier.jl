struct InverseBarrier <: BarrierMethod end

function barrierValue(::InverseBarrier, violation)
    any(v -> v >= 0, violation) && return Inf
    return sum(v -> -1/v, violation)
end