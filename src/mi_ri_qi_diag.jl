# Midiagonal, Ridiagonal and Qidiagonal matrices 


import LinearAlgebra: mul!, lmul!, ldiv!, \, /, *, inv, pinv, sqrt, adjoint
import Base: size, getindex, replace_in_print_matrix, rand, randn
export Ridiagonal, Qidiagonal, Midiagonal


"""
Q <: Qidiagonal ≡ QₙQₙ₋₁ ⋯ Q₂ is block bidiagonal on supdiagonal
R <: Ridiagonal ≡ R₂ ⋯ RₙRₙ₋₁ is block bidiagonal on subdiagonal
M <: Midiagonal ≡ block_diag(M₁, ⋯ ,Mₙ) is block diagonal

with Q.data == [Q₂, …, Qₙ₋₁, Qₙ] and 
with R.data == [R₂, …, Rₙ₋₁, Rₙ]
with M.data == [M₁, ⋯ ,Mₙ]
"""

struct Ridiagonal{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::Vector{M} # sub diagonal matrices stored along the last dimension
end

struct Qidiagonal{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::Vector{M} # sup diagonal matrices stored along the last dimension
end

struct Midiagonal{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::Vector{M} # diagonal matrices stored along the last dimension
end


Ridiagonal(A::Ridiagonal) = A
Qidiagonal(A::Qidiagonal) = A
Midiagonal(A::Midiagonal) = A

Ridiagonal{T}(A::Ridiagonal{T}) where {T} = A
Qidiagonal{T}(A::Qidiagonal{T}) where {T} = A
Midiagonal{T}(A::Midiagonal{T}) where {T} = A

Ridiagonal{T}(A::Ridiagonal) where {T} = Ridiagonal{T}(A.data)
Qidiagonal{T}(A::Qidiagonal) where {T} = Qidiagonal{T}(A.data)
Midiagonal{T}(A::Midiagonal) where {T} = Midiagonal{T}(A.data)

const RiQi{T,M} = Union{Ridiagonal{T,M}, Qidiagonal{T,M}}
const MiRiQi{T,M} = Union{Midiagonal{T,M}, Ridiagonal{T,M}, Qidiagonal{T,M}}

# ============================

sizes_from_blocksides(::Type{<:Midiagonal}, bs::Vector{Int}) = [(bs[i],bs[i]) for i=1:length(bs)]
sizes_from_blocksides(::Type{<:Ridiagonal}, bs::Vector{Int}) = [(bs[i+1],bs[i]) for i=1:length(bs)-1]
sizes_from_blocksides(::Type{<:Qidiagonal}, bs::Vector{Int}) = [(bs[i], bs[i+1]) for i=1:length(bs)-1]

nblocks(M::Midiagonal) = length(M.data)
nblocks(RQ::RiQi) = length(RQ.data)+1

diag_block_dlengths(M::Midiagonal) = size.(M.data, 1)

function diag_block_dlengths(Q::Qidiagonal)
    bs = size.(Q.data, 1)
    append!(bs, size(Q.data[end],2))
end 

function diag_block_dlengths(R::Ridiagonal)
    bs = size.(R.data, 2)
    append!(bs, size(R.data[end],1))
end 

# used to prep incoming arrays for lmul! or ldiv!
function _pblock_array(MRQ::MiRiQi, w::AbstractVector)
    bs  = diag_block_dlengths(MRQ)
    blocks(PseudoBlockArray(w, bs))
end

function _pblock_array(MRQ::MiRiQi, w::AbstractVector, v::AbstractVector)
    bs  = diag_block_dlengths(MRQ)
    blocks(PseudoBlockArray(w, bs)), blocks(PseudoBlockArray(v, bs))
end

rand(::Type{A}, bs::Vector{Int}) where {T,A<:Midiagonal{T}} = Midiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
rand(::Type{A}, bs::Vector{Int}) where {T,A<:Ridiagonal{T}} = Ridiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
rand(::Type{A}, bs::Vector{Int}) where {T,A<:Qidiagonal{T}} = Qidiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    

randn(::Type{A}, bs::Vector{Int}) where {T,A<:Midiagonal{T}} = Midiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
randn(::Type{A}, bs::Vector{Int}) where {T,A<:Ridiagonal{T}} = Ridiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    
randn(::Type{A}, bs::Vector{Int}) where {T,A<:Qidiagonal{T}} = Qidiagonal(map(sz -> rand(T,sz), sizes_from_blocksides(A, bs)))    


# AbstractMatrix methods
# =================

size(MRQ::MiRiQi)    = (n = sum(diag_block_dlengths(MRQ)); (n,n))
size(MRQ::MiRiQi, d) = d::Integer <= 2 ? size(MRQ)[d] : 1

function getindex(R::Ridiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(diag_block_dlengths(R))
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

function getindex(Q::Qidiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(diag_block_dlengths(Q))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]
    if Block4i  == Block4j
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        wbi == wbj ? one(T) : zero(T)
    elseif  Block4i + 1 == Block4j
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        Q.data[Block4i][wbi, wbj]
    else 
        return zero(T)
    end
end

function getindex(M::Midiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(diag_block_dlengths(M))
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
    row_or_col_Ix = blockedrange(diag_block_dlengths(MRQ))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]

    if i==j
        return s 
    elseif (MRQ isa Ridiagonal) && (Block4i == Block4j + 1)
        return s 
    elseif (MRQ isa Qidiagonal) && (Block4i + 1 == Block4j)
        return s
    elseif (MRQ isa Midiagonal) && (Block4i == Block4j)
        return s  
    else 
        return Base.replace_with_centered_mark(s)
    end
end

# lmul! (R₂ ⋯ RₙRₙ₋₁) * w or (Q'₂ ⋯ Q'ₙQ'ₙ₋₁) * w 
# =================

# R * w
function lmul!(R::Ridiagonal, w::AbstractVector)
    wbB = _pblock_array(R, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i+1], R.data[i], wbB[i], true, true)    
    end
    return w
end

# Q' * w
function lmul!(Qᴴ::Adjoint{<:Any, <:Qidiagonal}, w::AbstractVector)
    Q   = parent(Qᴴ)
    wbB = _pblock_array(Q, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i+1], Q.data[i]', wbB[i], true, true)    
    end
    return w
end


# lmul! (QₙQₙ₋₁ ⋯ Q₂)*w or (Rₙ'Rₙ₋₁' ⋯ R₂')*w
# =================

# Q * w
function lmul!(Q::Qidiagonal, w::AbstractVector)
    wbB = _pblock_array(Q, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i], Q.data[i], wbB[i+1], true, true)       
    end
    return w
end

# R' * w
function lmul!(Rᴴ::Adjoint{<:Any, <:Ridiagonal}, w::AbstractVector)
    R   = parent(Rᴴ)
    wbB = _pblock_array(R, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i], R.data[i]', wbB[i+1], true, true)       
    end
    return w
end

# ldiv! (R₂ ⋯ RₙRₙ₋₁)⁻¹ * w or (Q'₂ ⋯ Q'ₙQ'ₙ₋₁)⁻¹ * w
# =================

# inv(R) * w
function ldiv!(R::Ridiagonal, w::AbstractVector)
    wbB = _pblock_array(R, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i+1], R.data[i], wbB[i], -1, true)      
    end
    return w
end

# inv(Q)' * w
function ldiv!(Qᴴ::Adjoint{<:Any,<:Qidiagonal}, w::AbstractVector)
    Q   = parent(Qᴴ)
    wbB = _pblock_array(Q, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i+1], Q.data[i]', wbB[i], -1, true)      
    end
    return w
end



# ldiv! (QₙQₙ₋₁ ⋯ Q₂)⁻¹ * w or (Rₙ'Rₙ₋₁' ⋯ R₂')⁻¹ * w
# =================

# inv(R)' * w
function ldiv!(Rᴴ::Adjoint{<:Any,<:Ridiagonal}, w::AbstractVector)
    R   = parent(Rᴴ)
    wbB = _pblock_array(R, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i], R.data[i]', wbB[i+1], -1, true)     
    end
    return w
end

# inv(Q) * w
function ldiv!(Q::Qidiagonal, w::AbstractVector)
    wbB = _pblock_array(Q, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i], Q.data[i], wbB[i+1], -1, true)     
    end
    return w
end


# 3 arg mul!
# =================

function mul!(rw::AbstractVector, R::Union{RiQi, Adjoint{<:Any,<:RiQi}}, w::AbstractVector)
    copyto!(rw,w)
    lmul!(R, rw)
end

function mul!(rw::AbstractVector, M::Midiagonal, w::AbstractVector, α::Number, β::Number)
    rwB, wB = _pblock_array(M, rw, w)
    for i = 1:length(wB)
        mul!(rwB[i], M.data[i], wB[i], α, β)
    end
    return rw
end

# \ and * for Ridiagonal and Qidiagonal
# =========================

(*)(RQ::RiQi,                  w::AbstractVector) = lmul!(RQ, copy(w))
(*)(RQ::Adjoint{<:Any,<:RiQi}, w::AbstractVector) = lmul!(RQ, copy(w))

(\)(RQ::RiQi,                  w::AbstractVector) = ldiv!(RQ, copy(w))
(\)(RQ::Adjoint{<:Any,<:RiQi}, w::AbstractVector) = ldiv!(RQ, copy(w))

# special for Midiagonal
#  =========================

for op in (:*, :/, :\)
    @eval  $op(M::Midiagonal, w::AbstractVector) = $op(mortar(Diagonal(M.data)), w)
end

for op in (:inv, :pinv, :sqrt, :adjoint)
    @eval  $op(M::Midiagonal) = Midiagonal(map($op, M.data))
end
