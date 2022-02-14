# Chains of InvOrAdjOrVecc 
# =================================================

# `*` concatinates chains
# ------------------------------------

function *(O1::InvOrAdjOrVecc_VF, O2::InvOrAdjOrVecc_VF) 
    tuple(O1, O2)
end

function *(O1::AbstractMatrix, O2::InvOrAdjOrVecc_VF)
    tuple(O1, O2)
end

function *(O1::InvOrAdjOrVecc_VF, O2::AbstractMatrix)
    tuple(O1, O2)
end


## chains of ops concatonate

function *(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, O2::MatrixOrInvOrAdjOrVecc_VF) where N 
    Base.front(O1) * (Base.last(O1) * O2)
end

function *(O1::MatrixOrInvOrAdjOrVecc_VF, O2::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N
    (O1 * Base.first(O2)) * Base.tail(O2)
end

function *(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, O2::NTuple{M,MatrixOrInvOrAdjOrVecc_VF}) where {N,M}
    ## tuple(O1..., O2...) ## old version, slated for removal
    
    # Pull out the meeting endpoints to give them a chance to merge
    # Base.front(O1) * (Base.last(O1) * Base.first(O2)) * Base.tail(O2)
    Omid = Base.last(O1) * Base.first(O2) 
    if Omid isa Tuple
        return tuple(Base.front(O1)..., Omid..., Base.tail(O2)...)
    else
        return tuple(Base.front(O1)..., Omid, Base.tail(O2)...)
    end
end

# `\`
# ------------------------------------

function \(O1::InvOrAdjOrVecc_VF, O2::InvOrAdjOrVecc_VF)
    inv(O1) * O2
end

function \(O1::AbstractMatrix, O2::InvOrAdjOrVecc_VF)
    inv(O1) * O2
end

function \(O1::InvOrAdjOrVecc_VF, O2::AbstractMatrix)
    inv(O1) * O2
end


## chains of ops concatonate

function \(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, O2::MatrixOrInvOrAdjOrVecc_VF) where N
    inv(O1) * O2
end

function \(O1::MatrixOrInvOrAdjOrVecc_VF, O2::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N
    inv(O1) * O2
end

function \(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, O2::NTuple{M,MatrixOrInvOrAdjOrVecc_VF}) where {N,M}
    inv(O1) * O2
end

# `/`
# ------------------------------------

function /(O1::InvOrAdjOrVecc_VF, O2::InvOrAdjOrVecc_VF)
    O1 * inv(O2) 
end

function /(O1::AbstractMatrix, O2::InvOrAdjOrVecc_VF)
    O1 * inv(O2) 
end

function /(O1::InvOrAdjOrVecc_VF, O2::AbstractMatrix)
    O1 * inv(O2) 
end

## chains of ops concatonate

function /(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, O2::MatrixOrInvOrAdjOrVecc_VF) where N
    O1 * inv(O2)
end

function /(O1::MatrixOrInvOrAdjOrVecc_VF, O2::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N
    O1 * inv(O2)
end

function /(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, O2::NTuple{M,MatrixOrInvOrAdjOrVecc_VF}) where {N,M}
    O1 * inv(O2)
end


# `inv` broadcasts and reverses order
# ------------------------------------

function inv(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N
    tuple((inv(op) for op in reverse(O1))...)
end

pinv(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N = inv(O1)

# `adjoint` broadcasts and reverses order
function adjoint(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N
    tuple((adjoint(op) for op in reverse(O1))...)
end

# activate the lazy tuple when operating
# ------------------------------------

function *(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, f::AbstractVector) where N
    foldr(*, (O1..., f))
end

function \(O1::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}, f::AbstractVector) where N
    inv(O1) * f
end 

# ---------------------------------------------- 

function show(io::IO, ::MIME"text/plain", A::NTuple{N,MatrixOrInvOrAdjOrVecc_VF}) where N
    println(io, "$N - tuple of VecchiaFactors or Adjoints")
    println(io, typeof(A))
end




