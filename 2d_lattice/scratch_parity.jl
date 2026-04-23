using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# Run Trotter and TDVP GRAPE side-by-side at N=1 for a fixed iteration budget,
# compare final fidelities and per-iter wall time.

const MAX_ITER = 5

println("──── Baseline (Trotter) ────────────────────────────────")
include("MPS_GRAPE.jl")
t0 = time()
r_trott = run_grape(1; max_iter=MAX_ITER, save=false, verbose=true)
dt_trott = time() - t0
println("Trotter elapsed: $(round(dt_trott, digits=1)) s")

println()
println("──── New (MPO-TDVP) ────────────────────────────────────")
include("MPS_TDVP_GRAPE.jl")
t1 = time()
r_tdvp = run_grape_tdvp(1; max_iter=MAX_ITER, save=false, verbose=true)
dt_tdvp = time() - t1
println("TDVP elapsed: $(round(dt_tdvp, digits=1)) s")

println()
println("──── Parity summary ────────────────────────────────────")
println("Trotter final F = $(round(r_trott.fidelity, digits=6))")
println("TDVP    final F = $(round(r_tdvp.fidelity,   digits=6))")
println("|ΔF|            = $(round(abs(r_trott.fidelity - r_tdvp.fidelity), sigdigits=3))")
println("Trotter time    = $(round(dt_trott, digits=1)) s   (~$(round(dt_trott/MAX_ITER, digits=2)) s/iter)")
println("TDVP    time    = $(round(dt_tdvp,  digits=1)) s   (~$(round(dt_tdvp/MAX_ITER,  digits=2)) s/iter)")
println("Speedup         = $(round(dt_trott / max(dt_tdvp, 1e-6), digits=2))×")
