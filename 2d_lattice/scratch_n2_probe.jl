using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include("MPS_TDVP_GRAPE.jl")

# Warm-started N=2 TDVP GRAPE probe: 3 L-BFGS iters, no save.
# Goal: measure per-iter time, confirm warm-start path, check TDVP
# forward-pass fidelity under the Trotter N=1 pulse rescaled to N=2.
t0 = time()
r = run_grape_tdvp(2; max_iter=3, save=false, verbose=true)
println()
println("── Probe summary ────────────────────────────────────────")
@printf("  N=2   grad evals = ?   final F = %.6f   wall = %.1fs\n",
        r.fidelity, time() - t0)
