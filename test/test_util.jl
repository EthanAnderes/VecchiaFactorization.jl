using VecchiaFactorization
import VecchiaFactorization as VF
using Test

@testset "util.jl" begin 

    @test VF.block_split(20, 10) == [10,10]    
    @test VF.block_split(20, 5) == [5,5,5,5] 
    @test VF.block_split(20, 11) == [11,9]   
    @test VF.block_split(11, 11) == [11]   
    @test VF.block_split(11, 10) == [10,1]   

    blk_sizes = [100, 150, 20, 50]
    @inferred VF.mortar_Bidiagonal_fill(1.0, blk_sizes)
    @inferred VF.initalize_bidiag_lblks(typeof(1.0), blk_sizes)
    @inferred VF.mortar_Tridiagonal_fill(1.0, blk_sizes) ## getting error here

end


