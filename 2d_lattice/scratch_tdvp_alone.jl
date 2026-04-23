using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include("MPS_TDVP_GRAPE.jl")
t0 = time()
r = run_grape_tdvp(1; max_iter=10, save=false, verbose=true)
dt = time() - t0
println()
println("TDVP total wall time: $(round(dt, digits=1)) s")
println("Final F: $(round(r.fidelity, digits=6))")
