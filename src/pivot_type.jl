# Wrapper for permutations
# ============================================================

struct Piv <: VecchiaFactor{Bool}
    perm::Vector{Int}
    function Piv(p::AbstractVector{<:Integer})
        @assert isperm(p)
        new(collect(p))
    end
end

# Interface for VecchiaFactors
# ===================================

# base methods ldiv! and mul!
mul!(w::AbstractVector, p::Piv, v::AbstractVector) = (copyto!(w,v); lmul!(p,w))
lmul!(p::Piv, v::AbstractVector) = permute!(v, p)
ldiv!(p::Piv, v::AbstractVector) = invpermute!(v, p)

# bypass adjoint and inv
adjoint(p::Piv)   = Piv(invperm(p.perm))
inv(p::Piv)       = Piv(invperm(p.perm))

# merge products of Piv 
*(p::Piv, q::Piv) = Piv(p * q.perm)
\(p::Piv, q::Piv) = Piv(p \ q.perm)

# Non-Interface for VecchiaFactors
# ===================================

pinv(p::Piv)      = inv(p)
transpose(p::Piv) = inv(p)

size(p::Piv)    = (l = length(p.perm); (l,l))
size(p::Piv, d) = d::Integer <= 2 ? size(p)[d] : 1

permute!(v::AbstractVector, p::Piv)    = permute!(v, p.perm)
invpermute!(v::AbstractVector, p::Piv) = invpermute!(v, p.perm)

## if we turn these on then Piv * Diag -> sparse which 
## which effects the collaps of the op_chain for Beams. 
## Not sure if we want that.
# *(p::Piv, m::AbstractMatrix) = m[p.perm,:]
# *(m::AbstractMatrix, p::Piv) = m[:, invperm(p.perm)]
# \(p::Piv, m::AbstractMatrix) = pinv(p) * m
# \(m::AbstractMatrix, p::Piv) = pinv(m) * p
# /(p::Piv, m::AbstractMatrix) = p * pinv(m)
# /(m::AbstractMatrix, p::Piv) = m * pinv(p)

function getindex(p::Piv, i::Integer, j::Integer) 
    (p.perm[i] == j) ? true : false 
end
