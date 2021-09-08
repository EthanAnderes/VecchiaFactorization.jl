module VecchiaFactorization

using LinearAlgebra # BLAS.set_num_threads(1)
using BlockArrays: PseudoBlockArray, AbstractBlockMatrix, Block, blocks, blocksizes,
blockedrange, findblockindex, blockindex 
using BlockBandedMatrices: BlockDiagonal, BlockBidiagonal
using FillArrays
## using ArrayLayouts

export Vecchia, InvVecchia, VecchiaPivoted, InvVecchiaPivoted

include("ri_qi_diag.jl")
export Ridiagonal, Qidiagonal


# Make a Vecchia Struct
# ===============================

## represents cov Σ = invR * M * invR'
struct Vecchia{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Vector{RT}
    M::Vector{MT}
    bsds::Vector{Int} #blocksides
end
struct VecchiaPivoted{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Vector{RT}
    M::Vector{MT}
    bsds::Vector{Int} #blocksides
    piv::Vector{Int} # permutation
end

## For VecchiaPivoted the Vecchia approx is applied to Σ[piv,piv].
## so that Σ ≈ (invR * M * invR')[invperm(piv), invperm(piv)]
## and  Σ * v ≈ ((invR * M * invR') * v[piv])[invperm(piv)]

## represents inverse cov invΣ = R' * invM * R
struct InvVecchia{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Vector{RT}
    invM::Vector{MT}
    bsds::Vector{Int} # blocksides
end
struct InvVecchiaPivoted{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Vector{RT}
    invM::Vector{MT}
    bsds::Vector{Int} # blocksides
    piv::Vector{Int} # permutation
end

## For InvVecchiaPivoted the same pivot transform is applied
## and  Σ \ v ≈ ((R' * invM * R) * v[piv])[invperm(piv)]


InvVecc_or_Vecc{T} = Union{Vecchia{T}, InvVecchia{T}} where {T}
InvVecc_or_Vecc_Pivoted{T} = Union{VecchiaPivoted{T}, InvVecchiaPivoted{T}} where {T}

## length(bsds) = nblocks
## Block i has size bsds[i] × bsds[i]
## length(R) = length(M) - 1


# Constructor based on a BlockArray overlay
# ================================================

function Vecchia(;diag_blocks::Vector{DM}, subdiag_blocks::Vector{sDM}) where {DM<:AbstractMatrix, sDM<:AbstractMatrix}
	nblocks = length(diag_blocks)
	@assert length(subdiag_blocks) == nblocks - 1
	R = map(1:nblocks-1) do i 
		- subdiag_blocks[i] / diag_blocks[i]
	end 

	M = map(1:nblocks) do i 
		if i==1 
			return diag_blocks[i]
		else 
			## return diag_blocks[i] + R[i-1] * subdiag_blocks[i-1]'
			return diag_blocks[i] - subdiag_blocks[i-1] / diag_blocks[i-1] * subdiag_blocks[i-1]'
		end
	end

	bsn = map(b->size(b,1), diag_blocks)

	Vecchia(R, M, bsn)
end

function Vecchia(Σ::AbstractBlockMatrix)
	bsn     = blocksizes(Σ)[1] 
	bi      = Block.(1:length(bsn))
	Vecchia(;
		diag_blocks=[Σ[i,i] for i in bi], 
		subdiag_blocks=[Σ[i+1,i] for i in bi[1:end-1]],
	)
end

function Vecchia(Σ::Matrix, bsn::Vector{Int})
	n, m = size(Σ)
	@assert n == m == sum(bsn)
	Vecchia(PseudoBlockArray(Σ, bsn, bsn))
end


function VecchiaPivoted(V::Vecchia, piv::Vector{Int})
	VecchiaPivoted(V.R, V.M, V.bsds, piv)
end


function InvVecchiaPivoted(V::InvVecchia, piv::Vector{Int})
	InvVecchiaPivoted(V.R, V.invM, V.bsds, piv)
end

# Vecchia: internal methods for left mult
# ============================


## Σ = invR * M * invR'
function _vecclmul!(R::Vector{RT}, M::Vector{MT}, bsds::Vector{Int}, w::AbstractVector) where {T, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}}
	wbB = blocks(PseudoBlockArray(w, bsds))
	nb  = length(bsds)
	## z = inv(R)' * w
	for i in nb-1:-1:1
		mul!(wbB[i], R[i]', wbB[i+1], -1, true)		
	end
	## q = M * z
	for i in 1:nb	
		mul!(wbB[i], M[i], copy(wbB[i]))
	end
	## inv(R) * q
	for i in 1:nb-1
		mul!(wbB[i+1], R[i], wbB[i], -1, true)		
	end
	return w
end

function _vecclmul!(R::Vector{RT}, M::Vector{MT}, bsds::Vector{Int}, W::AbstractMatrix) where {T, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}}
	for i=1:size(W,2)
		_vecclmul!(R, M, bsds, view(W, :, i))
	end 
	return W 
end

## TODO: eventually make somethine like an R matrix type and Q matrix type (Q = R')
## for now I'm just making an internal method for it but eventually this will be
## removed ...
function _Rldiv!(R::Vector{RT}, bsds::Vector{Int}, w::AbstractVector) where {T, RT<:AbstractMatrix{T}}
	wbB = blocks(PseudoBlockArray(w, bsds))
	nb  = length(bsds)
	## inv(R) * q
	for i in 1:nb-1
		mul!(wbB[i+1], R[i], wbB[i], -1, true)		
	end
	return w
end

# InvVecchia: internal methods for left mult
# ============================

## invΣ = R' * invM * R
function _invvecclmul!(R::Vector{RT}, invM::Vector{MT}, bsds::Vector{Int}, w::AbstractVector) where {T, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}}
	wbB = blocks(PseudoBlockArray(w, bsds))
	nb  = length(bsds)
	## z = R * w
	for i in nb-1:-1:1
		mul!(wbB[i+1], R[i], wbB[i], true, true)	
	end
	## q = invM * z
	for i in 1:nb	
		mul!(wbB[i], invM[i], copy(wbB[i]))
	end
	## R' * q
	for i in 1:nb-1
		mul!(wbB[i], R[i]', wbB[i+1], true, true)		
	end
	return w
end


function _invvecclmul!(R::Vector{RT}, invM::Vector{MT}, bsds::Vector{Int}, W::AbstractMatrix) where {T, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}}
	for i=1:size(W,2)
		_invvecclmul!(R, invM, bsds, view(W, :, i))
	end 
	return W 
end



# Exported left mult methods
# ============================

Base.:*(V::InvVecc_or_Vecc,         W::Union{AbstractVector, AbstractMatrix}) = lmul!(V, copy(W))
Base.:*(V::InvVecc_or_Vecc_Pivoted, W::Union{AbstractVector, AbstractMatrix}) = lmul!(V, copy(W))

function LinearAlgebra.lmul!(V::Vecchia, W::Union{AbstractVector, AbstractMatrix})
	_vecclmul!(V.R, V.M, V.bsds, W)
end

function LinearAlgebra.lmul!(V::InvVecchia, W::Union{AbstractVector, AbstractMatrix})
	_invvecclmul!(V.R, V.invM, V.bsds, W)
end

function LinearAlgebra.lmul!(V::VecchiaPivoted, W::Union{AbstractVector, AbstractMatrix})
	for i = 1:size(W,2)
		_vecclmul!(V.R, V.M, V.bsds, permute!(view(W, :, i), V.piv))
		invpermute!(view(W, :, i), V.piv)
	end
	return W
end

function LinearAlgebra.lmul!(V::InvVecchiaPivoted, W::Union{AbstractVector, AbstractMatrix})
	for i = 1:size(W,2)
		_invvecclmul!(V.R, V.invM, V.bsds, permute!(view(W, :, i), V.piv))
		invpermute!(view(W, :, i), V.piv)
	end
	return W
end


# other LinearAlgebra methods
# ====================================

function Rmat(V::InvVecc_or_Vecc{T}) where {T}
	nb  = length(V.bsds)
	👀  = map(b->Matrix(Eye{T}(b)), V.bsds)
	BlockBidiagonal(👀, V.R, :L)
end

function Rᴴmat(V::InvVecc_or_Vecc{T}) where {T}
	nb  = length(V.bsds)
	👀  = map(b->Matrix(Eye{T}(b)), V.bsds)
	BlockBidiagonal(👀, map(x->Matrix(x'), V.R), :U)
end

# pinv(V)  and inv(V)
# ----------------------------

LinearAlgebra.pinv(V::Vecchia) = InvVecchia(deepcopy(V.R), map(pinv, V.M), V.bsds) 
LinearAlgebra.pinv(V::InvVecchia) = Vecchia(deepcopy(V.R), map(pinv, V.invM), V.bsds) 
Base.inv(V::InvVecc_or_Vecc) = pinv(V)

LinearAlgebra.pinv(V::VecchiaPivoted) = InvVecchiaPivoted(deepcopy(V.R), map(pinv, V.M), V.bsds, V.piv) 
LinearAlgebra.pinv(V::InvVecchiaPivoted) = VecchiaPivoted(deepcopy(V.R), map(pinv, V.invM), V.bsds, V.piv)
Base.inv(V::InvVecc_or_Vecc_Pivoted) = pinv(V)


# size 
# ------------------------------

Base.size(V::InvVecc_or_Vecc{T})         where {T} = (nrws = sum(V.bsds); (nrws,nrws))
Base.size(V::InvVecc_or_Vecc_Pivoted{T}) where {T} = (nrws = sum(V.bsds); (nrws,nrws))

Base.size(V::InvVecc_or_Vecc{T}, d)         where {T} = d::Integer <= 2 ? size(V)[d] : 1
Base.size(V::InvVecc_or_Vecc_Pivoted{T}, d) where {T} = d::Integer <= 2 ? size(V)[d] : 1

# Matrix(V) 
# ----------------------------


# Σ = invR * M * invR'
function Base.Matrix(V::Vecchia)
	M    = BlockDiagonal(V.M)
	invR = inv(Rmat(V))
	Matrix(invR  * M * invR')
end

# Σ = Pᵀ * invR * M * invR' * P
function Base.Matrix(Vᴾ::VecchiaPivoted)
	V      = Vecchia(Vᴾ.R, Vᴾ.M, Vᴾ.bsds)
	invpiv = invperm(Vᴾ.piv) 
	M      = BlockDiagonal(V.M)
	invR   = inv(Rmat(V))
	Matrix(invR  * M * invR')[invpiv, invpiv]
end

# invΣ = R' * invM * R
function Base.Matrix(iV::InvVecchia{T}) where {T}
	invM = BlockDiagonal(iV.invM)
	R    = Rmat(iV)
	Matrix(R' * invM * R)
end

# invΣ = Pᵀ * R' * invM * R * P
function Base.Matrix(iVᴾ::InvVecchiaPivoted{T}) where {T}
	iV      = InvVecchia(iVᴾ.R, iVᴾ.invM, iVᴾ.bsds)
	invpiv = invperm(iVᴾ.piv) 
	invM = BlockDiagonal(iV.invM)
	R    = Rmat(iV)
	Matrix(R' * invM * R)[invpiv, invpiv]
end


# inv_cholesky gives inv(cholesky(Σ)) -> LowerTriangular
# ----------------------------

function inv_cholesky(V::Vecchia)
	L⁻¹s = map(V.M) do M 
		## inv(cholesky(Hermitian(M, :L)).L)
		Matrix(inv(cholesky(Hermitian(M, :L)).L))
	end 
	## TODO try and use skyline block matrix for 
	BlockDiagonal(L⁻¹s) * Rmat(V)
end


## TODO, this needs work. 
## For one, need to add inv(cholesky(Σ)) -> pivoted LowerTriangular
## or something like that

end
