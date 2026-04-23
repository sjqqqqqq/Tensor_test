using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Printf, JLD2, LinearAlgebra
include("MPS_TDVP_GRAPE.jl")

# Quick probe: starting-F of the N=1 Trotter pulse replayed under N=2 TDVP
# dynamics, for several J-rescale factors. Answers whether warm-start is
# even feasible before we invest in full L-BFGS runs.

N = 2
T = 2π
nsteps = 100
dt = T / nsteps
cutoff = 1e-10
maxdim = 64

s = siteinds("SoftBoson", 4)
psi0, psi_target = default_states(s, N; cutoff, maxdim)

# Load the Trotter N=1 pulse.
trotter_file = joinpath(DATA_DIR, "GRAPE_2d_MPS_1.jld2")
d = load(trotter_file)
nprev = Int(d["n"]) - 1
@assert nprev == nsteps "nsteps mismatch"
prev = hcat([d[String(k)][1:nprev] for k in CTRL_KEYS]...)
println("Loaded Trotter N=1 pulse (F=$(round(d["fidelity"], digits=6))) from $(basename(trotter_file))")
println("Controls shape: ", size(prev))

function fidelity_at(ctrls)
    psis_end = forward_tdvp(ctrls, psi0, s; cutoff, maxdim, dt)[end]
    return abs2(inner(psi_target, psis_end))
end

factors = Dict("no_rescale (×1.0)"           => 1.0,
               "Rabi-match (×√(1/2))"        => sqrt(1/2),
               "inverse (×√2)"               => sqrt(2.0),
               "aggressive (×0.5)"           => 0.5,
               "aggressive (×2.0)"           => 2.0)

for (label, f) in pairs(factors)
    c = copy(prev)
    c[:, 8] .*= f         # Ja column
    c[:, 9] .*= f         # Jb column
    t0 = time()
    F = fidelity_at(c)
    @printf("  %-30s  F(0) = %.6f   (%.1fs)\n", label, F, time()-t0)
end
