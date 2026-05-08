# exact_factored.jl
# Replay a saved GRAPE pulse using the factored eigendecomposition step
# (eigendecompose only the Da×Da / Db×Db hopping factors; never form the
# D×D Kron matrices). Mirrors 2d_lattice/exact_time_evol.jl's structure.
# Outputs the fidelity trajectory F(t).
#
# Usage:
#   julia exact_factored.jl path/to/GRAPE_2d_Npair_N.jld2 [out.jld2]

using JLD2, LinearAlgebra, Printf

include(joinpath(@__DIR__, "..", "2d_lattice", "exact_GRAPE.jl"))

function fidelity_trajectory_factored(sys::System2DNpair, psi0, psi_target, ctrls, dt)
    nsteps = size(ctrls, 1)
    fid = zeros(Float64, nsteps)
    psi = Vector{ComplexF64}(psi0)
    fid[1] = abs2(dot(psi_target, psi))
    for n in 1:nsteps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n, :]
        psi = trotter_step2d(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
        fid[n+1] = abs2(dot(psi_target, psi))
    end
    return fid
end

function run(infile::String; outfile::Union{String,Nothing}=nothing)
    d = load(infile)
    N = haskey(d, "Npair") ? d["Npair"] : d["N"]
    n = d["n"]; T = d["T"]
    Va1,Va2,Va3 = d["Va1"], d["Va2"], d["Va3"]
    Vb1,Vb2,Vb3 = d["Vb1"], d["Vb2"], d["Vb3"]
    U, Ja, Jb   = d["U"],   d["Ja"],  d["Jb"]
    Fopt = get(d, "fidelity", NaN)
    ctrls = hcat(Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb)
    dt = T / (n - 1)

    sys = build_system(N)
    psi0, psi_target = default_states(sys)

    @printf("Factored eigendecomp replay: N=%d, Da=%d, Db=%d, D=%d, n=%d, T=%.4f\n",
            N, sys.Da, sys.Db, sys.D, n, T)
    @printf("Saved F = %s\n", isnan(Fopt) ? "n/a" : string(round(Fopt, digits=6)))

    t0 = time()
    fid = fidelity_trajectory_factored(sys, psi0, psi_target, ctrls, dt)
    wall = time() - t0
    @printf("Recomputed F(0) = %.6f, F(T) = %.6f   wall = %.3f s\n",
            fid[1], fid[end], wall)

    t = collect(range(0, T, length=n))
    out = isnothing(outfile) ?
          joinpath(@__DIR__, "data", "exact_factored_N$(N).jld2") :
          outfile
    mkpath(dirname(out))
    jldsave(out; N=N, T=T, n=n, dt=dt, t=t, fid=fid, F_final=fid[end], wall=wall)
    println("Saved $out")
    return fid
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) ≥ 1 || error("usage: julia exact_factored.jl <input.jld2> [output.jld2]")
    run(ARGS[1]; outfile = length(ARGS) ≥ 2 ? ARGS[2] : nothing)
end
