struct QuadraticPenalty <: PenaltyMethod end

function penaltyValue(::QuadraticPenalty, violation::Real)
    return violation^2
end

function penaltyValue(::QuadraticPenalty, violation::AbstractVector)
    return sum(v^2 for v in violation)
end