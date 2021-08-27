module VecchiaFactorization

using LinearAlgebra # BLAS.set_num_threads(1)
using BlockArrays: PseudoBlockArray, blocks
using BlockBandedMatrices: BlockDiagonal, BlockBidiagonal
using FillArrays
## using ArrayLayouts

export Vecchia, InvVecchia

# Make a Vecchia Struct
# ===============================

## represents cov Σ = invR * M * invR'
struct Vecchia{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Vector{RT}
    M::Vector{MT}
    bsds::Vector{Int} #blocksides
end

## represents inverse cov invΣ = R' * invM * R
struct InvVecchia{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Vector{RT}
    invM::Vector{MT}
    bsds::Vector{Int} # blocksides
end

InvVecc_or_Vecc{T} = Union{Vecchia{T}, InvVecchia{T}} where {T}

## length(bsds) = nblocks
## Block i has size bsds[i] × bsds[i]
## length(R) = length(M) - 1

# left mult 
# ============================

Base.:*(V::InvVecc_or_Vecc, w::AbstractVector) = lmul!(V, copy(w))

## Σ = invR * M * invR'
function LinearAlgebra.lmul!(V::Vecchia, w::AbstractVector)
	wbB = blocks(PseudoBlockArray(w, V.bsds))
	nb  = length(V.bsds)
	## z = inv(V.R)' * w
	for i in nb-1:-1:1
		mul!(wbB[i], V.R[i]', wbB[i+1], -1, true)		
	end
	## q = V.M * z
	for i in 1:nb	
		mul!(wbB[i], V.M[i], copy(wbB[i]))
	end
	## inv(V.R) * q
	for i in 1:nb-1
		mul!(wbB[i+1], V.R[i], wbB[i], -1, true)		
	end
	return w
end

## invΣ = R' * invM * R
function LinearAlgebra.lmul!(iV::InvVecchia, w::AbstractVector)
	wbB = blocks(PseudoBlockArray(w, iV.bsds))
	nb  = length(iV.bsds)
	## z = iV.R * w
	for i in nb-1:-1:1
		mul!(wbB[i+1], iV.R[i], wbB[i], true, true)	
	end
	## q = iV.invM * z
	for i in 1:nb	
		mul!(wbB[i], iV.invM[i], copy(wbB[i]))
	end
	## iV.R' * q
	for i in 1:nb-1
		mul!(wbB[i], iV.R[i]', wbB[i+1], true, true)		
	end
	return w
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

LinearAlgebra.pinv(V::InvVecchia) = Vecchia(deepcopy(V.R), map(pinv, V.M), V.bsds) 

Base.inv(V::InvVecc_or_Vecc) = pinv(V)

# size 
# ------------------------------

Base.size(V::InvVecc_or_Vecc{T}) where {T} = (nrws = sum(V.bsds); (nrws,nrws))

Base.size(V::InvVecc_or_Vecc{T}, d) where {T} = d::Integer <= 2 ? size(V)[d] : 1

# Matrix(V) 
# ----------------------------

# invΣ = R' * invM * R
function Base.Matrix(iV::InvVecchia{T}) where {T}
	invM = BlockDiagonal(iV.invM)
	R    = Rmat(iV)
	R' * invM * R
end

# Σ = invR * M * invR'
function Base.Matrix(V::Vecchia)
	M    = BlockDiagonal(V.M)
	invR = inv(Rmat(V))
	invR  * M * invR'
end

# inv_cholesky gives inv(cholesky(Σ)) -> LowerTriangular
# ----------------------------

function inv_cholesky(V::Vecchia)
	L⁻¹s = map(cholesky(Hermitian(V.M))) do M 
		## inv(cholesky(Hermitian(M, :L)).L)
		Matrix(inv(cholesky(Hermitian(M, :L)).L))
	end 
	## TODO try and use skyline block matrix for 
	BlockDiagonal(L⁻¹s) * Rmat(V)
end

end
