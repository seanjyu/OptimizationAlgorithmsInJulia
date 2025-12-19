struct L1Penalty <: PenaltyMethod end

function penaltyValue(::L1Penalty, violation::Real)
    return abs(violation)
end

function penaltyValue(::L1Penalty, violation::AbstractVector)
    return sum(abs(v) for v in violation)
end

struct SmoothL1Penalty <: PenaltyMethod
    delta::Float64
end

SmoothL1Penalty(; delta=1.0) = SmoothL1Penalty(delta)

function penaltyValue(p::SmoothL1Penalty, violation::Real)
    v = abs(violation)
    if v <= p.delta
        return v^2 / (2 * p.delta)
    else
        return v - p.delta / 2
    end
end

function penaltyValue(p::SmoothL1Penalty, violation::AbstractVector)
    return sum(penaltyValue(p, v) for v in violation)
end