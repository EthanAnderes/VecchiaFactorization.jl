# constructor for the vecchia factorization approxiamtion

function vecchia(Σ::AbstractMatrix{T}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}) where T
	R, M, P = R_M_P(Σ, blk_sizes, perm)
    println("Vecchia Factorization approximation: Pᵀ R⁻¹ M R⁻ᴴ P")
	return P' * inv(R) * M * inv(R)' * P
end

function vecchia(Σ::AbstractMatrix{T}, blk_sizes::AbstractVector{<:Integer}) where T
	R, M, P = R_M_P(Σ, blk_sizes)
    println("Vecchia Factorization approximation: R⁻¹ M R⁻ᴴ")
	return inv(R) * M * inv(R)'
end

function R_M_P(Σ::AbstractMatrix{T}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes)) where T
	LinearAlgebra.checksquare(Σ)
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	M = Vector{Matrix{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			M[ic] = Σ[blk_indices[ic], blk_indices[ic]]
		else 
			U 		= cholesky(Σ[blk_indices[ic-1], blk_indices[ic-1]]).U 
			C 		= Σ[blk_indices[ic], blk_indices[ic-1]] / U
			R[ic-1] = - C / U'
			M[ic]   = Σ[blk_indices[ic], blk_indices[ic]] - C*C'
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end




