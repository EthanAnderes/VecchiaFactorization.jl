# lazy wrapper types for inverse and adjoint view of a VecchiaFactor
# ============================================================

struct Adj{T,S} <: VecchiaFactor{T}
    parent::S
    function Adj{T,S}(A::S) where {T,S}
        new(A)
    end
end

struct Inv{T,S} <: VecchiaFactor{T}
    parent::S
    function Inv{T,S}(A::S) where {T,S}
        new(A)
    end
end

const Adj_VF            = Adj{<:Any, <:VecchiaFactor}
const Inv_VF            = Inv{<:Any, <:VecchiaFactor}
const Inv_Adj_VF        = Inv{<:Any, Adj{<:Any, <:VecchiaFactor}}
const InvOrAdj_VF       = Union{Inv_VF, Adj_VF}
const InvOrAdjOrVecc_VF = Union{VecchiaFactor, Adj_VF, Inv_VF, Inv_Adj_VF}
const MatrixOrInvOrAdjOrVecc_VF = Union{AbstractMatrix, InvOrAdjOrVecc_VF}

# -----------------------

Adj(A) = Adj{eltype(A),typeof(A)}(A)
function adjoint(A::VecchiaFactor) 
    # form the adjoint wrapper
    Adj(A)
end
function adjoint(A::Adj)
    # adjoint cancels Adj
    A.parent
end
function adjoint(A::Inv) 
    # always bring Inv out front
    Inv(adjoint(A.parent))
end 
function adjoint(A::Inv_Adj_VF) 
    # A.parent.parent is the base operator
    # Since the adjoint cancels Adj, we just need an inv of the base operator
    Inv(A.parent.parent)
end 

# -----------------------

Inv(A) = Inv{eltype(A),typeof(A)}(A)
function inv(A::VecchiaFactor)
    # form the inv wrapper
    Inv(A)
end
function inv(A::Inv)
    # inv of an Inv cancels
    A.parent
end
function inv(A::Adj)
    # Inv stays out front
    Inv(A)
end
function inv(A::Inv_Adj_VF) 
    # strip the outer Inv since it gets killed by inv
    A.parent
end

