import LinearAlgebra: *, \, /, inv, pinv, adjoint

# Chains of VeccFactorOrAdjoint 
# =================================================


function show(io::IO, ::MIME"text/plain", A::NTuple{N,VeccFactorOrAdjoint}) where N
    println(io, "$N - tuple of VecchiaFactors or Adjoints")
    println(io, typeof(A))
end

# `*` concatinates chains
# ------------------------------------

function *(O1::VeccFactorOrAdjoint{T}, O2::VeccFactorOrAdjoint{S}) where {T,S}
    tuple(O1, O2)
end

function *(O1::NTuple{N,VeccFactorOrAdjoint}, O2::VeccFactorOrAdjoint) where N 
    Base.front(O1) * (Base.last(O1) * O2)
end

function *(O1::VeccFactorOrAdjoint, O2::NTuple{N,VeccFactorOrAdjoint}) where N
    (O1 * Base.first(O2)) * Base.tail(O2)
end

function *(O1::NTuple{N,VeccFactorOrAdjoint}, O2::NTuple{M,VeccFactorOrAdjoint}) where {N,M}
    tuple(O1..., O2...)
end

# `inv` broadcasts and reverses order
# ------------------------------------

function inv(O1::NTuple{N,VeccFactorOrAdjoint}) where N
    tuple((inv(op) for op in reverse(O1))...)
end

pinv(O1::NTuple{N,VeccFactorOrAdjoint}) where N = inv(O1)

# `adjoint` broadcasts and reverses order
function adjoint(O1::NTuple{N,VeccFactorOrAdjoint}) where N
    tuple((adjoint(op) for op in reverse(O1))...)
end

# `\`
# ------------------------------------

function \(O1::VeccFactorOrAdjoint{T}, O2::VeccFactorOrAdjoint{S}) where {T,S}
    inv(O1) * O2
end

function \(O1::NTuple{N,VeccFactorOrAdjoint}, O2::VeccFactorOrAdjoint) where N
    inv(O1) * O2
end

function \(O1::VeccFactorOrAdjoint, O2::NTuple{N,VeccFactorOrAdjoint}) where N
    inv(O1) * O2
end

function \(O1::NTuple{N,VeccFactorOrAdjoint}, O2::NTuple{M,VeccFactorOrAdjoint}) where {N,M}
    inv(O1) * O2
end

# `/`
# ------------------------------------

function /(O1::VeccFactorOrAdjoint{T}, O2::VeccFactorOrAdjoint{S}) where {T,S}
    O1 * inv(O2) 
end

function /(O1::NTuple{N,VeccFactorOrAdjoint}, O2::VeccFactorOrAdjoint) where N
    O1 * inv(O2)
end

function /(O1::VeccFactorOrAdjoint, O2::NTuple{N,VeccFactorOrAdjoint}) where N
    O1 * inv(O2)
end

function /(O1::NTuple{N,VeccFactorOrAdjoint}, O2::NTuple{M,VeccFactorOrAdjoint}) where {N,M}
    O1 * inv(O2)
end

# activate the lazy tuple when operating
# ------------------------------------

function *(O1::NTuple{N,VeccFactorOrAdjoint}, f::AbstractVector) where {N}
    foldr(*, (O1..., f))
end

function \(O1::NTuple{N,VeccFactorOrAdjoint}, f::AbstractVector) where {N}
    inv(O1) * f
end 






