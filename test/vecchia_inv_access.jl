using LinearAlgebra
using VecchiaFactorization
import VecchiaFactorization as VF
using BlockArrays
using Test
using LBblocks

@testset "vecchia_inv_access.jl" begin 

    n = 320 

    A = @sblock let θtru = 1.5, n
        K = (x,y,θtru) -> exp(- θtru * abs(x - y) ^ 0.8 )
        x = range(0,4,n)
        K.(x, x', θtru)
    end

    B = @sblock let θtru = 0.75, n
        K = (x,y,θtru) -> exp(- θtru * abs(x - y) ^ 1.5 )
        x = range(0,4,n)
        K.(x, x', θtru)
    end

    block_sizes = [100, 150, 20, 50]

    R̄, M̄, = VF.R_M_P(A, block_sizes)
    R̃, M̃, = VF.R_M_P(B, block_sizes)

    R, M, M⁻¹ =  VF.vecchia_add_inv(R̄, M̄, R̃, M̃)  

    Ǎ⁻¹ = Matrix(R̄' * inv(M̄) * R̄)
    B̌⁻¹ = Matrix(R̃' * inv(M̃) * R̃)
    Č⁻¹ = Matrix(R' * M⁻¹ * R)
    @test Č⁻¹ ≈ Ǎ⁻¹ + B̌⁻¹ 

end
