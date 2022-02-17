using LinearAlgebra
using VecchiaFactorization
import VecchiaFactorization as VF
using Test
using LBblocks


@testset "vecchia_inv_access.jl: instantiate_inv! and instantiate_inv" begin 

    n = 750 

    A = @sblock let θtru = 1.5, n
        K = (x,y,θtru) -> exp(- θtru * abs(x - y) ^ 0.8 )
        x = range(0,4,n)
        K.(x, x', θtru)
    end

    block_sizes = [200, 250, 200, 100]
    perm = vcat(1:n÷2, n:-1:n÷2+1)
    R, M, P = VF.R_M_P(A, block_sizes, perm)

    X = similar(A)
    VF.instantiate_inv!(X, R, M, P)

    X′_pre_perm = VF.instantiate_inv_tridiagonal(R, M)
    X′ = X′_pre_perm[P.perm, P.perm]

    @test X ≈ X′

    invA_v = Matrix(P' * R' * inv(M) * R * P)
    @test invA_v ≈ X

    #=
    using BenchmarkTools
    @benchmark VF.instantiate_inv!(X, $R, $M, $P)
    @benchmark VF.instantiate_inv!(X, $R, $M)
    @benchmark VF.instantiate_inv_tridiagonal($R, $M)

    @benchmark VF.instantiate_inv!(X, $R, $M, $P)
    @benchmark Matrix(VF.instantiate_inv_tridiagonal($R, $M))[P.perm, P.perm]

    @benchmark VF.instantiate_inv!(X, $R, $M)
    @benchmark Matrix(VF.instantiate_inv_tridiagonal($R, $M))



    =#

end



@testset "vecchia_inv_access.jl: vecchia_add_inv" begin 

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


    T1 = Diagonal(rand(n))
    T2 = rand(n,n)
    T3 = cholesky(T2 * T2').L
    v = rand(n) 
    
    @test (R * M * T1) * v ≈ (R * M) * (T1 * v) 
    @test (R * M * T2) * v ≈ (R * M) * (T2 * v) 
    @test (R * M * T3) * v ≈ (R * M) * (T3 * v)

    @test (T1 * R * M) * v ≈ T1 * (R * M * v) 
    @test (T2 * R * M) * v ≈ T2 * (R * M * v) 

    @test (inv(R') * M * T2) * v ≈ (inv(R') * M) * (T2 * v) 
    @test (T2 * inv(R') * M) * v ≈ T2 * (inv(R') * M * v)


end


