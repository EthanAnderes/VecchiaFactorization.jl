
import LinearAlgebra: mul!, lmul!, ldiv!, \, /, *, inv, pinv, adjoint, transpose
import Base: size, permute!, invpermute!, getindex, replace_in_print_matrix
export Piv

# Wrapper for permutations
# ============================================================

struct Piv <: VecchiaFactor{Bool}
    perm::Vector{Int}
    function Piv(p::Vector{Int})
        @assert sort(p) == 1:length(p)
        new(p)
    end
end

# Hook into 
#--------------------------------

*(p::Piv, v::VeccFactorOrAdjoint) = tuple(p, v)
*(v::VeccFactorOrAdjoint, p::Piv) = tuple(v, p)
\(p::Piv, v::VeccFactorOrAdjoint) = inv(p) * v
\(v::VeccFactorOrAdjoint, p::Piv) = inv(v) * p

#--------------------------------

inv(p::Piv)       = Piv(invperm(p.perm))
pinv(p::Piv)      = inv(p)
adjoint(p::Piv)   = inv(p)
transpose(p::Piv) = inv(p)

size(p::Piv)    = (l = length(p.perm); (l,l))
size(p::Piv, d) = d::Integer <= 2 ? size(p)[d] : 1

permute!(v::AbstractVector, p::Piv)    = permute!(v, p.perm)
invpermute!(v::AbstractVector, p::Piv) = invpermute!(v, p.perm)

*(p::Piv, q::Piv) = Piv(p * q.perm)
\(p::Piv, q::Piv) = Piv(p \ q.perm)

mul!(w::AbstractVector, p::Piv, v::AbstractVector) = (copyto!(w,v); lmul!(p,w))
lmul!(p::Piv, v::AbstractVector) = permute!(v, p)
ldiv!(p::Piv, v::AbstractVector) = invpermute!(v, p)

*(p::Piv, v::AbstractVector) = lmul!(p, copy(v))
\(p::Piv, v::AbstractVector) = ldiv!(p, copy(v))

*(p::Piv, m::AbstractMatrix) = m[p.perm,:]
*(m::AbstractMatrix, p::Piv) = m[:, invperm(p.perm)]
\(p::Piv, m::AbstractMatrix) = pinv(p) * m
\(m::AbstractMatrix, p::Piv) = pinv(m) * p
/(p::Piv, m::AbstractMatrix) = p * pinv(m)
/(m::AbstractMatrix, p::Piv) = m * pinv(p)

*(p::Piv, m::Adjoint{<:Any, <:AbstractMatrix}) = adjoint(m.parent * adjoint(p))
*(m::Adjoint{<:Any, <:AbstractMatrix}, p::Piv) = adjoint(adjoint(p) * m.parent)
\(p::Piv, m::Adjoint{<:Any, <:AbstractMatrix}) = pinv(p) * m
\(m::Adjoint{<:Any, <:AbstractMatrix}, p::Piv) = pinv(m) * p
/(p::Piv, m::Adjoint{<:Any, <:AbstractMatrix}) = p * pinv(m)
/(m::Adjoint{<:Any, <:AbstractMatrix}, p::Piv) = m * pinv(p)

function getindex(p::Piv, i::Integer, j::Integer) 
    (p.perm[i] == j) ? true : false 
end

function replace_in_print_matrix(p::Piv, i::Integer, j::Integer, s::AbstractString) 
    p[i,j] ? s : Base.replace_with_centered_mark(s)
end