
@testset "vecchia_approx.jl and sparse_matrix_show.jl" begin 

    n = 320 

    Σ = @sblock let θtru = 1.5, n
        K = (x,y,θtru) -> exp(- θtru * abs(x - y) ^ 0.8 )
        x = range(0,4,n)
        K.(x, x', θtru)
    end

    block_sizes = [100, 150, 20, 50]
    perm = sortperm(@. sin((1:n)*2*π/n))

    R1, M1, P1 = VF.R_M_P(Σ, block_sizes, perm)
    R2, M2, P2 = VF.R_M_P(Σ[perm, perm], block_sizes)
    R3, M3, P3 = VF.R_M_P((i,j) -> Σ[i,j], block_sizes, perm)

    @test VF.sparse(R1) ≈ VF.sparse(R2) ≈ VF.sparse(R3)
    @test VF.sparse(M1) ≈ VF.sparse(M2) ≈ VF.sparse(M3)
    @test VF.sparse(P1) ≈ VF.sparse(P3)

    @test VF.Matrix(R1) ≈ VF.Matrix(R2) ≈ VF.Matrix(R3)
    @test VF.Matrix(M1) ≈ VF.Matrix(M2) ≈ VF.Matrix(M3)
    @test VF.Matrix(P1) ≈ VF.Matrix(P3)

    @test VF.sparse(R1') ≈ VF.sparse(R1)';
    @test VF.sparse(M1') ≈ VF.sparse(M1)';
    @test VF.sparse(P1') ≈ VF.sparse(P1)';

    VΣ = VF.vecchia(Σ, block_sizes, perm)
    VF.sparse(inv(VΣ))
    VF.sparse(inv(adjoint(VΣ)))
    VF.sparse(adjoint(inv(VΣ)))

    VF.Matrix(VΣ)
    VF.Matrix(inv(VΣ))
    VF.Matrix(adjoint(VΣ))
    VF.Matrix(inv(adjoint(VΣ)))
    VF.Matrix(adjoint(inv(VΣ)))

end


