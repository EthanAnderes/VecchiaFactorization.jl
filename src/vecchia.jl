# Vecchia and InvVecchia with pivoted versions of both
# ============================================================

## represents cov Σ = invR * M * invR'
struct Vecchia{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Ridiagonal{T,RT}
    M::Midiagonal{T,MT}
    bsds::Vector{Int}
end
struct VecchiaPivoted{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Ridiagonal{T,RT}
    M::Midiagonal{T,MT}
    bsds::Vector{Int}
    piv::Vector{Int} # permutation
end

# For VecchiaPivoted the Vecchia approx is applied to Σ[piv,piv].
# so that Σ ≈ (invR * M * invR')[invperm(piv), invperm(piv)]
# and  Σ * v ≈ ((invR * M * invR') * v[piv])[invperm(piv)]

# represents inverse cov invΣ = R' * invM * R
struct InvVecchia{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Ridiagonal{T,RT}
    invM::Midiagonal{T,MT}
    bsds::Vector{Int}
end
struct InvVecchiaPivoted{T<:Number, RT<:AbstractMatrix{T}, MT<:AbstractMatrix{T}} <: Factorization{T}
    R::Ridiagonal{T,RT}
    invM::Midiagonal{T,MT}
    bsds::Vector{Int}
    piv::Vector{Int} # permutation
end

# For InvVecchiaPivoted the same pivot transform is applied
# and  Σ \ v ≈ ((R' * invM * R) * v[piv])[invperm(piv)]

InvVecc_or_Vecc{T}         = Union{Vecchia{T}, InvVecchia{T}} where {T}
InvVecc_or_Vecc_Pivoted{T} = Union{VecchiaPivoted{T}, InvVecchiaPivoted{T}} where {T}

# length(bsds) = nblocks
# Block i has size bsds[i] × bsds[i]
# length(R) = length(M) - 1

# Constructor based on a BlockArray overlay
# ================================================

function Vecchia(;diag_blocks::Vector{DM}, subdiag_blocks::Vector{sDM}) where {DM<:AbstractMatrix, sDM<:AbstractMatrix}
	nblocks = length(diag_blocks)
	@assert length(subdiag_blocks) == nblocks - 1
	R = map(1:nblocks-1) do i
		if sum(abs2, diag_blocks[i]) == 0
			return diag_blocks[i]
		else
			return - subdiag_blocks[i] / diag_blocks[i]
		end
	end |> Ridiagonal

	M = map(1:nblocks) do i 
		if (i==1) | (sum(abs2, diag_blocks[i]) == 0)
			return diag_blocks[i]
		else 
			# return diag_blocks[i] + R.data[i-1] * subdiag_blocks[i-1]'
			return diag_blocks[i] - subdiag_blocks[i-1] / diag_blocks[i-1] * subdiag_blocks[i-1]'
		end
	end |> Midiagonal

	Vecchia(R, M, diag_block_dlengths(M))
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


# left mult methods, non-pivoted
# ============================

# Σ = invR * M * invR'
function *(V::Vecchia, w::AbstractVector)
	rw = copy(w)
	ldiv!(V.R', rw) 
	mul!(rw, V.M, copy(rw)) 
	ldiv!(V.R, rw) 
end

# invΣ = R' * invM * R
function *(V::InvVecchia, w::AbstractVector)
	rw = copy(w)
	lmul!(V.R, rw) 
	mul!(rw, V.invM, copy(rw)) 
	lmul!(V.R', rw) 
end

# Σ' = invR * M' * invR'
function *(Vᴴ::Adjoint{<:Any,<:Vecchia}, w::AbstractVector)
	V  = parent(Vᴴ)
	rw = copy(w)
	ldiv!(V.R', rw) 
	mul!(rw, V.M', copy(rw)) 
	ldiv!(V.R, rw) 
end

# invΣ' = R' * invM' * R
function *(Vᴴ::Adjoint{<:Any,<:InvVecchia}, w::AbstractVector)
	V  = parent(Vᴴ)
	rw = copy(w)
	lmul!(V.R, rw) 
	mul!(rw, V.invM', copy(rw)) 
	lmul!(V.R', rw) 
end

# left mult methods, pivoted
# ============================

# Σ = P' * invR * M * invR' * P
function *(V::VecchiaPivoted, w::AbstractVector)
	V′ = Vecchia(V.R, V.M, V.bsds)
	w′ = permute!(copy(w), V.piv)
	invpermute!(V′ * w′, V.piv)
end

# invΣ = P' * R' * invM * R * P
function *(V::InvVecchiaPivoted, w::AbstractVector)
	V′ = InvVecchia(V.R, V.invM, V.bsds)
	w′ = permute!(copy(w), V.piv)
	invpermute!(V′ * w′, V.piv)
end

# Σ' = P' * invR * M' * invR' * P
function *(Vᴴ::Adjoint{<:Any,<:VecchiaPivoted}, w::AbstractVector)
	V  = parent(Vᴴ)
	V′ = Vecchia(V.R, V.M, V.bsds)
	w′ = permute!(copy(w), V.piv)
	invpermute!(V′' * w′, V.piv)
end

# invΣ' = P' * R' * invM' * R * P
function *(Vᴴ::Adjoint{<:Any,<:InvVecchiaPivoted}, w::AbstractVector)
	V  = parent(Vᴴ)
	V′ = InvVecchia(V.R, V.invM, V.bsds)
	w′ = permute!(copy(w), V.piv)
	invpermute!(V′' * w′, V.piv)
end

# other LinearAlgebra methods
# ====================================

function Rmat(V::InvVecc_or_Vecc{T}) where {T}
	nb  = length(V.bsds)
	👀  = map(b->Matrix(Eye{T}(b)), V.bsds)
	BlockBidiagonal(👀, V.R.data, :L)
end

function Rᴴmat(V::InvVecc_or_Vecc{T}) where {T}
	nb  = length(V.bsds)
	👀  = map(b->Matrix(Eye{T}(b)), V.bsds)
	BlockBidiagonal(👀, map(x->Matrix(x'), V.R.data), :U)
end

# pinv(V)  and inv(V)
# ----------------------------

inv(V::InvVecc_or_Vecc) = pinv(V)

pinv(V::Vecchia)           = InvVecchia(deepcopy(V.R), pinv(V.M), V.bsds)
pinv(V::VecchiaPivoted)    = InvVecchiaPivoted(deepcopy(V.R), pinv(V.M), V.bsds, V.piv) 
pinv(V::InvVecchia)        = Vecchia(deepcopy(V.R), pinv(V.invM), V.bsds)
pinv(V::InvVecchiaPivoted) = VecchiaPivoted(deepcopy(V.R), pinv(V.invM), V.bsds, V.piv)

inv(V::InvVecc_or_Vecc_Pivoted) = pinv(V)

# size 
# ------------------------------

size(V::InvVecc_or_Vecc{T})         where {T} = (nrws = sum(V.bsds); (nrws,nrws))
size(V::InvVecc_or_Vecc_Pivoted{T}) where {T} = (nrws = sum(V.bsds); (nrws,nrws))

size(V::InvVecc_or_Vecc{T}, d)         where {T} = d::Integer <= 2 ? size(V)[d] : 1
size(V::InvVecc_or_Vecc_Pivoted{T}, d) where {T} = d::Integer <= 2 ? size(V)[d] : 1

# Matrix(V) 
# ----------------------------

# Σ = invR * M * invR'
function Matrix(V::Vecchia)
	M    = BlockDiagonal(V.M.data)
	invR = inv(Rmat(V))
	Matrix(invR  * M * invR')
end

# Σ = Pᵀ * invR * M * invR' * P
function Matrix(Vᴾ::VecchiaPivoted)
	V      = Vecchia(Vᴾ.R, Vᴾ.M, Vᴾ.bsds)
	invpiv = invperm(Vᴾ.piv) 
	M      = BlockDiagonal(V.M.data)
	invR   = inv(Rmat(V))
	Matrix(invR  * M * invR')[invpiv, invpiv]
end

# invΣ = R' * invM * R
function Matrix(iV::InvVecchia{T}) where {T}
	invM = BlockDiagonal(iV.invM.data)
	R    = Rmat(iV)
	Matrix(R' * invM * R)
end

# invΣ = Pᵀ * R' * invM * R * P
function Matrix(iVᴾ::InvVecchiaPivoted{T}) where {T}
	iV      = InvVecchia(iVᴾ.R, iVᴾ.invM, iVᴾ.bsds)
	invpiv = invperm(iVᴾ.piv) 
	invM = BlockDiagonal(iV.invM.data)
	R    = Rmat(iV)
	Matrix(R' * invM * R)[invpiv, invpiv]
end


# inv_cholesky gives inv(cholesky(Σ)) -> LowerTriangular
# ----------------------------



# function inv_cholesky(V::Vecchia)
# 	L⁻¹s = map(V.M.data) do M 
# 		# inv(cholesky(Hermitian(M, :L)).L)
# 		Matrix(inv(cholesky(Hermitian(M, :L)).L))
# 	end 
# 	# TODO try and use skyline block matrix for 
# 	BlockDiagonal(L⁻¹s) * Rmat(V)
# end

# TODO, this needs work. 
# For one, need to add inv(cholesky(Σ)) -> pivoted LowerTriangular
# or something like that
