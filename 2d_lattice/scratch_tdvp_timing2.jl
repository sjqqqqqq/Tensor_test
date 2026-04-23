using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, Printf
include("MPS_TDVP_GRAPE.jl")

s = siteinds("SoftBoson", 4)
psi0, psi_target = default_states(s, 1; cutoff=1e-10, maxdim=64)

Random.seed!(42)
nsteps = 10; T = 2π; dt = T/100
ctrls = 0.3 .* randn(nsteps, 9)
ctrls[:,7] .+= 1.0; ctrls[:,8] .+= 1.0; ctrls[:,9] .+= 1.0

# Per-step timing, including MPO build
let psi = copy(psi0)
    for n in 1:nsteps
        t0 = time()
        H_n = build_H_step(@view(ctrls[n,:]), s)
        t1 = time()
        psi = tdvp_step(H_n, dt, psi; cutoff=1e-10, maxdim=64)
        t2 = time()
        @printf("step %2d  MPO=%.3fs  tdvp=%.3fs  link=%d  F=%.4f\n",
                n, t1-t0, t2-t1, maxlinkdim(psi), abs2(inner(psi_target, psi)))
        flush(stdout)
    end
end
