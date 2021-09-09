
import Base: size, parent, showarg
import LinearAlgebra: adjoint, inv, pinv, *

# lazy wrapper type for inverse view of Ridiagonal or 
# Qidiagonal. This allows defining a chain of operators 
# Ridiagonal, Qidiagonal and their inverses. 

# Note: much of this follows /julia/adjtrans.jl in structure and syntax

struct Inv{T,S} <: AbstractMatrix{T}
    parent::S
    function Inv{T,S}(A::S) where {T,S}
        ## checkeltype_adjoint(T, eltype(A))
        new(A)
    end
end

Inv(A) = Adjoint{eltype(A),typeof(A)}(A)

inv(A::Inv)  = A.parent

pinv(A::Inv) = A.parent

adjoint(A::Inv) = Inv(adjoint(A.parent))

parent(A::Inv) = A.parent

size(A::Inv) = reverse(size(A.parent))

function showarg(io::IO, v::Inv, toplevel)
    print(io, "inv(")
    Base.showarg(io, parent(v), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(v))
end

*(A::Inv, v::AbstractVector) = A \ v
