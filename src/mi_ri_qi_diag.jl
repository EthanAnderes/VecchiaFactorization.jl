# Midiagonal, Ridiagonal and Qidiagonal matrices 
# ============================================================

"""
Q <: Qidiagonal ≡ QₙQₙ₋₁ ⋯ Q₂ is block bidiagonal on supdiagonal
R <: Ridiagonal ≡ R₂ ⋯ RₙRₙ₋₁ is block bidiagonal on subdiagonal
M <: Midiagonal ≡ block_diag(M₁, ⋯ ,Mₙ) is block diagonal

with Q.data == [Q₂, …, Qₙ] and 
with R.data == [R₂, …, Rₙ]
with M.data == [M₁, ⋯ ,Mₙ]
"""

struct Ridiagonal{T,M<:AbstractMatrix{T}} <: VecchiaFactor{T}
    data::Vector{M} # sub diagonal matrices stored along the last dimension
end

struct Qidiagonal{T,M<:AbstractMatrix{T}} <: VecchiaFactor{T}
    data::Vector{M} # sup diagonal matrices stored along the last dimension
end

struct Midiagonal{T,M<:AbstractMatrix{T}} <: VecchiaFactor{T}
    data::Vector{M} # diagonal matrices stored along the last dimension
end

const RiQi{T,M}      = Union{Ridiagonal{T,M}, Qidiagonal{T,M}}
const MiRiQi{T,M}    = Union{Midiagonal{T,M}, Ridiagonal{T,M}, Qidiagonal{T,M}}

# Interface for VecchiaFactors
# ===================================

# bypass adjoint and inv
inv(M::Midiagonal)      = Midiagonal(map(inv, M.data)) 
adjoint(M::Midiagonal)  = Midiagonal(map(adjoint, M.data)) 

# merge products of Midiagonals
*(M1::Midiagonal, M2::Midiagonal) = Midiagonal(map(*, M1.data, M2.data))


# ldiv!
# ---------------------------------

# inv(R) * w ≡ (R₂ ⋯ RₙRₙ₋₁)⁻¹ * w
function ldiv!(R::Ridiagonal, w::AbstractVector{T}) where T
    wbB = block_array(R, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i+1], R.data[i], wbB[i], -1, true)      
    end
    return w
end

# inv(Q)' * w ≡ (Q'₂ ⋯ Q'ₙQ'ₙ₋₁)⁻¹ * w
# function ldiv!(Qᴴ::Adjoint{<:Any,<:Qidiagonal}, w::AbstractVector{T}) where T
#     Q   = parent(Qᴴ)
#     wbB = block_array(Q, w)
#     for i in 1:length(wbB)-1
#         mul!(wbB[i+1], Q.data[i]', wbB[i], -1, true)      
#     end
#     return w
# end

# inv(R)' * w ≡ (Rₙ'Rₙ₋₁' ⋯ R₂')⁻¹ * w
function ldiv!(Rᴴ::Adjoint{<:Any,<:Ridiagonal}, w::AbstractVector{T}) where T
    R   = parent(Rᴴ)
    wbB = block_array(R, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i], R.data[i]', wbB[i+1], -1, true)     
    end
    return w
end

# inv(Q) * w ≡ (QₙQₙ₋₁ ⋯ Q₂)⁻¹ * w 
# function ldiv!(Q::Qidiagonal, w::AbstractVector{T}) where T
#     wbB = block_array(Q, w)
#     for i in length(wbB)-1:-1:1
#         mul!(wbB[i], Q.data[i], wbB[i+1], -1, true)     
#     end
#     return w
# end

# M * w
function ldiv!(M::A, w::AbstractVector)  where {A<:Midiagonal}
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        copyto!(rwB[i], M.data[i] \ wB[i])
    end
    return rw
end

# inv(M') * w
function ldiv!(Mᴴ::A, w::AbstractVector) where {A<:Adjoint{<:Any,<:Midiagonal}}
    M = parent(Mᴴ)
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        copyto!(rwB[i], M.data[i]' \ wB[i])
    end
    return rw
end


# 3 and 5 arg mul! 
# ---------------------------------

function mul!(rw::AbstractVector, RQ::A, w::AbstractVector) where {A<:Union{RiQi, Adjoint{<:Any,<:RiQi}}}
    copyto!(rw,w)
    lmul!(RQ, rw)
end

function mul!(rw::AbstractVector, M::A, w::AbstractVector) where {A<:Union{Midiagonal, Adjoint{<:Any,<:Midiagonal}}}
    mul!(rw, M, w, true, false)
end

function mul!(rw::AbstractVector, M::A, w::AbstractVector, α::Number, β::Number) where {A<:Midiagonal}
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        mul!(rwB[i], M.data[i], wB[i], α, β)
    end
    return rw
end

function mul!(rw::AbstractVector, Mᴴ::A, w::AbstractVector, α::Number, β::Number) where {A<:Adjoint{<:Any,<:Midiagonal}}
    M = parent(Mᴴ)
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        mul!(rwB[i], M.data[i]', wB[i], α, β)
    end
    return rw
end

# mul! above call down to lmul! below

# R * w ≡ (R₂ ⋯ RₙRₙ₋₁) * w
function lmul!(R::Ridiagonal, w::AbstractVector{T}) where T
    wbB = block_array(R, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i+1], R.data[i], wbB[i], true, true)    
    end
    return w
end

# Q' * w ≡ (Q'₂ ⋯ Q'ₙQ'ₙ₋₁) * w 
# function lmul!(Qᴴ::Adjoint{<:Any, <:Qidiagonal}, w::AbstractVector{T}) where T
#     Q   = parent(Qᴴ)
#     wbB = block_array(Q, w)
#     for i in length(wbB)-1:-1:1
#         mul!(wbB[i+1], Q.data[i]', wbB[i], true, true)    
#     end
#     return w
# end

# Q * w ≡ (QₙQₙ₋₁ ⋯ Q₂) * w
# function lmul!(Q::Qidiagonal, w::AbstractVector{T}) where T
#     wbB = block_array(Q, w)
#     for i in 1:length(wbB)-1
#         mul!(wbB[i], Q.data[i], wbB[i+1], true, true)       
#     end
#     return w
# end

# R' * w ≡ (Rₙ'Rₙ₋₁' ⋯ R₂')*w
function lmul!(Rᴴ::Adjoint{<:Any, <:Ridiagonal}, w::AbstractVector{T}) where T
    R   = parent(Rᴴ)
    wbB = block_array(R, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i], R.data[i]', wbB[i+1], true, true)       
    end
    return w
end


# Non-interface methods
# ================================================

Ridiagonal(A::Ridiagonal) = A
# Qidiagonal(A::Qidiagonal) = A
Midiagonal(A::Midiagonal) = A

Ridiagonal{T}(A::Ridiagonal{T}) where {T} = A
# Qidiagonal{T}(A::Qidiagonal{T}) where {T} = A
Midiagonal{T}(A::Midiagonal{T}) where {T} = A

Ridiagonal{T}(A::Ridiagonal) where {T} = Ridiagonal{T}(A.data)
# Qidiagonal{T}(A::Qidiagonal) where {T} = Qidiagonal{T}(A.data)
Midiagonal{T}(A::Midiagonal) where {T} = Midiagonal{T}(A.data)


sizes_from_blocksides(::Type{<:Midiagonal}, bs::Vector{Int}) = [(bs[i],bs[i]) for i=1:length(bs)]
sizes_from_blocksides(::Type{<:Ridiagonal}, bs::Vector{Int}) = [(bs[i+1],bs[i]) for i=1:length(bs)-1]
# sizes_from_blocksides(::Type{<:Qidiagonal}, bs::Vector{Int}) = [(bs[i], bs[i+1]) for i=1:length(bs)-1]

nblocks(M::Midiagonal) = length(M.data)
nblocks(RQ::RiQi) = length(RQ.data)+1



block_size(M::Midiagonal, d) = 0 < d::Integer <= 2 ? size.(M.data, 1) : 1

function block_size(R::Ridiagonal, d)
    if 0 < d::Integer <= 2
        bs = size.(R.data, 2)
        append!(bs, size(R.data[end],1))
    else 
        return 1 
    end
end 

# function block_size(Q::Qidiagonal, d)
#     if 0 < d::Integer <= 2
#         bs = size.(Q.data, 1)
#         append!(bs, size(Q.data[end],2))
#         return bs 
#     else 
#         return 1 
#     end
# end 



# used to prep incoming arrays for lmul! or ldiv!
function block_array(MRQ::MiRiQi, w::AbstractVector)
    bs  = block_size(MRQ,2)
    blocks(PseudoBlockArray(w, bs))
end

function block_array(MRQ::MiRiQi, w::AbstractVector, v::AbstractVector)
    bs  = block_size(MRQ,2)
    blocks(PseudoBlockArray(w, bs)), blocks(PseudoBlockArray(v, bs))
end

# Do we really need these?
rand(::Type{A}, bs::Vector{Int}) where {T,A<:Midiagonal{T}} = Midiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
rand(::Type{A}, bs::Vector{Int}) where {T,A<:Ridiagonal{T}} = Ridiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
# rand(::Type{A}, bs::Vector{Int}) where {T,A<:Qidiagonal{T}} = Qidiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    

randn(::Type{A}, bs::Vector{Int}) where {T,A<:Midiagonal{T}} = Midiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
randn(::Type{A}, bs::Vector{Int}) where {T,A<:Ridiagonal{T}} = Ridiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
# randn(::Type{A}, bs::Vector{Int}) where {T,A<:Qidiagonal{T}} = Qidiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    


# sqrt(M::Midiagonal)      = Midiagonal(map(sqrt, M.data)) 
# Matrix(M::Midiagonal)    = Midiagonal(map(Matrix, M.data)) 
# Hermitian(M::Midiagonal) = Midiagonal(map(x->Hermitian(x), M.data)) 
# Symmetric(M::Midiagonal) = Midiagonal(map(x->Symmetric(x), M.data)) 
# cholesky(M::Midiagonal)  = Midiagonal(map(x->cholesky(x).L, M.data)) 



# Conver Midiagonal, Ridiagonal to sparse arrays
# =================

sparse(M::Midiagonal) = sparse(mortar(Diagonal(M.data)))

function sparse(R::Ridiagonal{T}) where {T}
    n           = size(R,1)
    block_sizes = block_size(R,1)
    Rmat = PseudoBlockArray(
        spdiagm(fill(one(T),n)), 
        block_sizes, 
        block_sizes
    )
    for i=1:length(R.data)
        Rmat[Block(i+1,i)] .= R.data[i] 
    end
    sparse(Rmat)
end

# function sparse(Q::Qidiagonal{T}) where {T}
#     n           = size(Q,1)
#     block_sizes = block_size(Q,1)
#     Qmat = PseudoBlockArray(
#         spdiagm(fill(one(T),n)), 
#         block_sizes, 
#         block_sizes
#     )
#     for i=1:length(Q.data)
#         Qmat[Block(i,i+1)] .= Q.data[i] 
#     end
#     sparse(Qmat)
# end


Matrix(MRQ::MiRiQi) = Matrix(sparse(MRQ))



# AbstractMatrix methods
# =================

size(MRQ::MiRiQi)    = (n = sum(block_size(MRQ,1)); (n,n))
size(MRQ::MiRiQi, d) = d::Integer <= 2 ? size(MRQ)[d] : 1

function getindex(R::Ridiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(block_size(R,1))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]
    if Block4i  == Block4j
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        wbi == wbj ? one(T) : zero(T)
    elseif  Block4i == Block4j + 1
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        R.data[Block4j][wbi, wbj]
    else 
        return zero(T)
    end
end

# function getindex(Q::Qidiagonal{T}, i::Integer, j::Integer) where T
#     row_or_col_Ix = blockedrange(block_size(Q,1))
#     fbi = findblockindex(row_or_col_Ix, i)
#     fbj = findblockindex(row_or_col_Ix, j)
#     Block4i = fbi.I[1]
#     Block4j = fbj.I[1]
#     if Block4i  == Block4j
#         wbi = blockindex(fbi)
#         wbj = blockindex(fbj)
#         wbi == wbj ? one(T) : zero(T)
#     elseif  Block4i + 1 == Block4j
#         wbi = blockindex(fbi)
#         wbj = blockindex(fbj)
#         Q.data[Block4i][wbi, wbj]
#     else 
#         return zero(T)
#     end
# end

function getindex(M::Midiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(block_size(M,1))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]
    if Block4i == Block4j
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        M.data[Block4i][wbi, wbj]
    else 
        return zero(T)
    end
end

function replace_in_print_matrix(MRQ::MiRiQi, i::Integer, j::Integer, s::AbstractString)
    row_or_col_Ix = blockedrange(block_size(MRQ,1))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]
    if (MRQ isa Ridiagonal) && (Block4i == Block4j + 1)
        return s 
    elseif (MRQ isa Qidiagonal) && (Block4i + 1 == Block4j)
        return s
    elseif (MRQ isa Midiagonal) && (Block4i == Block4j)
        return s  
    else 
        return Base.replace_with_centered_mark(s)
    end
end
