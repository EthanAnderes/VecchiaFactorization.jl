using LinearAlgebra
## BLAS.set_num_threads(1)
using VecchiaFactorization
import VecchiaFactorization as VF
using BlockArrays
# using Random
using Test
using LBblocks

@testset "test_mi_ri.jl" begin 

    @sblock let 

        bs = [2, 4, 3]
        @test VF.sizes_from_blocksides(Ridiagonal, bs) == [(4,2), (3,4)]
        @test VF.sizes_from_blocksides(Midiagonal, bs) == [(2,2), (4,4), (3,3)]
        
        Ri = randn(Ridiagonal{Float64}, bs)
        Mi = randn(Midiagonal{Float64}, bs)

        @test Ri isa Ridiagonal
        @test Mi isa Midiagonal

        @test VF.block_size(Ri,1) == [2,4,3]
        @test VF.block_size(Mi,1) == [2,4,3]
        
        @test size(Ri) == (9,9)
        @test size(Mi) == (9,9)
        
        @test (size(Ri,1), size(Ri,2)) == size(Ri)
        @test (size(Mi,1), size(Mi,2)) == size(Mi)

    end


    @sblock let 
        ## bs = vcat(fill(60,7), fill(80,10))
        ## bs = fill(80,20)
        bs = [2, 4, 3, 10]
        ## bs = fill(10,20)
        ## bs = fill(10,200)

        Ri = randn(Ridiagonal{Float64}, bs)
        Mi = randn(Midiagonal{Float64}, bs)

        v = rand(Float64, sum(bs))

        @inferred Ri * v
        @inferred Ri' * v
        @inferred Ri \ v 
        @test (Ri \ v) ≈ ldiv!(Ri, copy(v)) rtol = 1e-10

        ## -----
        x = 1:size(Ri,1)
        v = zeros(Float64, length(x)); 
        for i = 1:2  # it is strange how unstable this inversion is 
            v[rand(x)] = 1
        end
        ## -----
        # 
        ## τ = 10; v = sin.(τ .* 2 .* π .* x ./ x[end])
        ## -----
        ## v = rand(Float64, sum(bs))
        ## -----
        v1 = Ri \ (Ri * v)
        w1 = Ri * (Ri \ v)
        @test v ≈ v1 rtol=1e-5
        @test v ≈ w1 rtol=1e-5

    end

    #=
    using BenchmarkTools

    bs = fill(50,20)
    T  = ComplexF64        
    Ri = randn(Ridiagonal{T}, bs)
    Mi = randn(Midiagonal{T}, bs)
    Mi′ = map(Hermitian, Mi.data) |> Midiagonal

    t = ComplexF64
    v  = randn(t, size(Ri,1))
    w  = randn(t, size(Ri,1))
    M = randn(t, size(Ri,1), size(Ri,1))
    @benchmark mul!(w, Mi, v, true, false)  # 9 μs
    @benchmark Mi * v  # 9 μs
    @benchmark Ri * v  # 9 μs
    @benchmark mul!(w, Mi′, v, true, false)  # 9 μs
    @benchmark Mi′ * v  # 9 μs

    @benchmark Mi' * v # 9 μs
    @benchmark Ri' * v # 9 μs

    @benchmark Mi \ v  # 716.999 μs
    @benchmark Ri \ v  # 9 μs

    @benchmark Mi' \ v # 734 μs
    @benchmark Ri' \ v # 9 μs

    @benchmark M * v # 112 μs
    =#

end
