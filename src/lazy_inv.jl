# lazy wrapper type for inverse view of a VecchiaFactor
# ============================================================

struct Inv{T,S} <: AbstractMatrix{T}
    parent::S
    function Inv{T,S}(A::S) where {T,S}
        ## checkeltype_adjoint(T, eltype(A))
        new(A)
    end
end

Inv(A) = Inv{eltype(A),typeof(A)}(A)

parent(A::Inv)  = A.parent
inv(A::Inv)     = A.parent
adjoint(A::Inv) = Inv(adjoint(A.parent))
size(A::Inv)    = reverse(size(A.parent))

# these eventually get called out to ldiv! and mul!
# TODO: do we need these?
# *(A::Inv{T,S}, v::AbstractVector{W}) where {T,S,W} = A.parent \ v 
\(A::Inv{T,S}, v::AbstractVector{W}) where {T,S,W} = A.parent * v

function show(io::IO, ::MIME"text/plain", A::Inv)
    print(io, typeof(A), "\n", A.parent)
end

