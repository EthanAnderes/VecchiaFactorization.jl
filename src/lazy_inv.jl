import Base: size, parent, show
import LinearAlgebra: adjoint, inv, pinv, *, \
export Inv

# lazy wrapper type for inverse view of a VecchiaFactor
# ============================================================

struct Inv{T,S} <: VecchiaFactor{T}
    parent::S
    function Inv{T,S}(A::S) where {T,S}
        ## checkeltype_adjoint(T, eltype(A))
        new(A)
    end
end

Inv(A) = Inv{eltype(A),typeof(A)}(A)

parent(A::Inv{T,S})  where {T,S} = A.parent
inv(A::Inv{T,S})     where {T,S} = A.parent
pinv(A::Inv{T,S})    where {T,S} = A.parent
adjoint(A::Inv{T,S}) where {T,S} = Inv(adjoint(A.parent))
size(A::Inv{T,S})    where {T,S} = reverse(size(A.parent))
*(A::Inv{T,S}, v::AbstractVector{W}) where {T,S,W} = A.parent \ v
\(A::Inv{T,S}, v::AbstractVector{W}) where {T,S,W} = A.parent * v

function show(io::IO, ::MIME"text/plain", A::Inv)
    print(io, typeof(A), "\n", A.parent)
end
