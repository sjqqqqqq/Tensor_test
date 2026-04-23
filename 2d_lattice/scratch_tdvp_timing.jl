using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, Printf
include("MPS_TDVP_GRAPE.jl")

s = siteinds("SoftBoson", 4)
psi0, psi_target = default_states(s, 1; cutoff=1e-10, maxdim=64)
H_nt = build_per_channel_mpos(s)
H_list = H_list_from(H_nt)

Random.seed!(42)
nsteps = 100; T = 2π; dt = T/nsteps
ctrls = 0.3 .* randn(nsteps, 9)
ctrls[:,7] .+= 1.0; ctrls[:,8] .+= 1.0; ctrls[:,9] .+= 1.0

# Warmup: 1 tdvp step to trigger JIT
println("warmup step...")
H1 = build_H_step(@view(ctrls[1,:]), s)
@time psi1 = tdvp_step(H1, dt, psi0; cutoff=1e-10, maxdim=64)
@time psi1 = tdvp_step(H1, dt, psi0; cutoff=1e-10, maxdim=64)

println("\nfull forward (100 steps):")
@time psis = forward_tdvp(ctrls, psi0, s; cutoff=1e-10, maxdim=64, dt=dt)
println("  F(end) = ", abs2(inner(psi_target, psis[end])))

println("\ninner(chi', H, psi) timing (9 calls):")
chi = copy(psi_target)
@time for k in 1:9; inner(chi', H_list[k], psis[50]); end
@time for k in 1:9; inner(chi', H_list[k], psis[50]); end

println("\nbackward pull 100 steps timing:")
chi = copy(psi_target)
@time for n in nsteps:-1:1
    H_n = build_H_step(@view(ctrls[n,:]), s)
    chi = tdvp_step(H_n, -dt, chi; cutoff=1e-10, maxdim=64)
end
