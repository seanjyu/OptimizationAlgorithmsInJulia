include("../../src/OptAlgos.jl")
using .OptAlgos
using Test

"""
Test convergence criteria function
"""

@testset "convergence test" begin
    c = ConvergenceCriteria()
    check, res = CheckConvergence(c, 0.1, 0, 0, 0, 0, 0)
    @test check == false
    check1, res1 = CheckConvergence(c, 1e-8, 0, 0, 0, 0, 0)
    @test check1 == true

    c1 = ConvergenceCriteria(xTol = 0.1)
    check2, res2 = CheckConvergence(c, 0, 0, 0, 1, 0, 0)
    @test check2 == true
end;