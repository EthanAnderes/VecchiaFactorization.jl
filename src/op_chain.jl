# Chains of InvOrAdjOrVecc 
# =================================================


# `*` concatinates chains
# ------------------------------------

# Note: the following argument types are unionall...allowing bipass for custom merge
function *(O1::InvOrAdjOrVecc, O2::InvOrAdjOrVecc) # testing
    tuple(O1, O2)
end

function *(O1::NTuple{N,InvOrAdjOrVecc}, O2::InvOrAdjOrVecc) where N 
    Base.front(O1) * (Base.last(O1) * O2)
end

function *(O1::InvOrAdjOrVecc, O2::NTuple{N,InvOrAdjOrVecc}) where N
    (O1 * Base.first(O2)) * Base.tail(O2)
end

function *(O1::NTuple{N,InvOrAdjOrVecc}, O2::NTuple{M,InvOrAdjOrVecc}) where {N,M}
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

# `inv` broadcasts and reverses order
# ------------------------------------

function inv(O1::NTuple{N,InvOrAdjOrVecc}) where N
    tuple((inv(op) for op in reverse(O1))...)
end

pinv(O1::NTuple{N,InvOrAdjOrVecc}) where N = inv(O1)

# `adjoint` broadcasts and reverses order
function adjoint(O1::NTuple{N,InvOrAdjOrVecc}) where N
    tuple((adjoint(op) for op in reverse(O1))...)
end

# `\`
# ------------------------------------

function \(O1::A, O2::InvOrAdjOrVecc) where A<:InvOrAdjOrVecc 
    inv(O1) * O2
end
# the second argument type below is unionall 
# ... this allows you to bipass for a specific InvOrAdjOrVecc

function \(O1::NTuple{N,InvOrAdjOrVecc}, O2::InvOrAdjOrVecc) where N
    inv(O1) * O2
end

function \(O1::InvOrAdjOrVecc, O2::NTuple{N,InvOrAdjOrVecc}) where N
    inv(O1) * O2
end

function \(O1::NTuple{N,InvOrAdjOrVecc}, O2::NTuple{M,InvOrAdjOrVecc}) where {N,M}
    inv(O1) * O2
end

# `/`
# ------------------------------------

function /(O1::A, O2::InvOrAdjOrVecc) where A<:InvOrAdjOrVecc 
    O1 * inv(O2) 
end
# the second argument type below is unionall 
# ... this allows you to bipass for a specific InvOrAdjOrVecc


function /(O1::NTuple{N,InvOrAdjOrVecc}, O2::InvOrAdjOrVecc) where N
    O1 * inv(O2)
end

function /(O1::InvOrAdjOrVecc, O2::NTuple{N,InvOrAdjOrVecc}) where N
    O1 * inv(O2)
end

function /(O1::NTuple{N,InvOrAdjOrVecc}, O2::NTuple{M,InvOrAdjOrVecc}) where {N,M}
    O1 * inv(O2)
end

# activate the lazy tuple when operating
# ------------------------------------

# function mul!(rf::AbstractVector, O1::NTuple{N,InvOrAdjOrVecc}, f::AbstractVector) where N
#     copyto!(rf, O1 * f) # is there a better way to do this?
# end

function *(O1::NTuple{N,InvOrAdjOrVecc}, f::AbstractVector) where N
    foldr(*, (O1..., f))
end

function \(O1::NTuple{N,InvOrAdjOrVecc}, f::AbstractVector) where N
    inv(O1) * f
end 

# function ldiv!(O1::NTuple{N,InvOrAdjOrVecc}, f::AbstractVector) where N
#     mul!(f, inv(O1), copy(f))
# end

# ---------------------------------------------- 

function show(io::IO, ::MIME"text/plain", A::NTuple{N,InvOrAdjOrVecc}) where N
    println(io, "$N - tuple of VecchiaFactors or Adjoints")
    println(io, typeof(A))
end




