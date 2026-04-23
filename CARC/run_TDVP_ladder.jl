using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# ─────────────────────────────────────────────────────────────────────────────
# Warm-start N=1 → 2 → ... → NMAX_TARGET ladder using MPO-TDVP GRAPE.
# Each step warm-starts from the previous TDVP pulse (or the Trotter baseline
# `data/GRAPE_2d_MPS_{N-1}.jld2` on the first rung if TDVP file is missing).
#
# Usage:
#   julia --project=. CARC/run_TDVP_ladder.jl [TARGET_N [MAX_ITER]]
# defaults: TARGET_N=10, MAX_ITER=50
#
# Sets SOFTBOSON_NMAX automatically so the local Hilbert dim can fit N pairs.
# ─────────────────────────────────────────────────────────────────────────────

target_N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
max_iter = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 50

# Set NMAX before include so the SoftBoson site dim is large enough.
# (NMAX is parsed from ENV at include time in soft_boson.jl.)
ENV["SOFTBOSON_NMAX"] = string(target_N)

include(joinpath(@__DIR__, "..", "2d_lattice", "MPS_TDVP_GRAPE.jl"))

println()
println("━"^70)
println("TDVP warm-start ladder, target N = $target_N, per-rung max_iter = $max_iter")
println("━"^70)

results = Dict{Int,Any}()
for N in 1:target_N
    println("\n══════ rung N = $N ══════")
    results[N] = run_grape_tdvp(N; max_iter=max_iter, save=true, verbose=true)
end

println()
println("━"^70)
println("Ladder summary:")
for N in 1:target_N
    @printf("  N=%2d : F = %.6f\n", N, results[N].fidelity)
end
