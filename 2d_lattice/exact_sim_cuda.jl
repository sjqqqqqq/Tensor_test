# exact_sim_cuda.jl
# GPU port of exact_sim.jl.  Loads a saved GRAPE pulse and simulates it with
# the exact (factored Trotter) propagator on the GPU.  Only the per-step state
# vector and the small evecs/evals/diag arrays live on the device; the controls
# and the fidelity trajectory stay on the host (just 9 + N_steps floats).
#
# Speedup vs. CPU comes from CUBLAS-backed GEMMs inside apply_a_cuda /
# apply_b_cuda.  For N=15 (Da = 816) the per-step work is ~4 GEMMs of size
# 816 × 816 × 816 complex; on an Nvidia GPU each Trotter step is dominated by
# kernel launch + memcpy of pa/pb, not compute.

using JLD2
using Plots
using LinearAlgebra
using CUDA
using Printf

include(joinpath(@__DIR__, "exact_GRAPE_cuda.jl"))

function fidelity_trajectory_cuda(N::Int, ctrls, T::Float64, num_steps::Int)
    sys = build_system_cuda(N)
    psi0, psi_tgt = default_states_cuda(sys)
    dt = T / (num_steps - 1)

    fid = zeros(Float64, num_steps)
    psi = copy(psi0)
    fid[1] = abs2(dot(psi_tgt, psi))
    for n in 1:num_steps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = trotter_step_cuda(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
        fid[n+1] = abs2(dot(psi_tgt, psi))
    end
    return fid
end

function plot_case_cuda(N::Int; file::String = "CARC/data/GRAPE_2d_Npair_$(N).jld2",
                        tag::String = "cuda")
    if !CUDA.functional()
        error("CUDA is not functional in this Julia session.")
    end
    @printf("CUDA device: %s\n", CUDA.name(CUDA.device()))

    d = load(file)
    n = d["n"]; T = d["T"]
    Va1, Va2, Va3 = d["Va1"], d["Va2"], d["Va3"]
    Vb1, Vb2, Vb3 = d["Vb1"], d["Vb2"], d["Vb3"]
    U, Ja, Jb = d["U"], d["Ja"], d["Jb"]
    Fopt = d["fidelity"]

    ctrls = hcat(Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb)
    t = range(0, T, length=n)

    println("N=$N:  n=$n, T=$T, saved F = ", round(Fopt, digits=6))
    elapsed = @elapsed fid = fidelity_trajectory_cuda(N, ctrls, T, n)
    @printf("   recomputed F(T) = %.6f   (%.2f s on GPU)\n", fid[end], elapsed)

    pVa = plot(t, Va1; label="Va1", lw=1.2, xlabel="t", ylabel="Va", title="a-type potentials")
    plot!(pVa, t, Va2; label="Va2", lw=1.2)
    plot!(pVa, t, Va3; label="Va3", lw=1.2)

    pVb = plot(t, Vb1; label="Vb1", lw=1.2, xlabel="t", ylabel="Vb", title="b-type potentials")
    plot!(pVb, t, Vb2; label="Vb2", lw=1.2)
    plot!(pVb, t, Vb3; label="Vb3", lw=1.2)

    pU = plot(t, U; label="U", lw=1.5, color=:black,
              xlabel="t", ylabel="U", title="On-site interaction", legend=false)

    pJ = plot(t, Ja; label="Ja", lw=1.5, color=:red,
              xlabel="t", ylabel="J", title="Hopping amplitudes")
    plot!(pJ, t, Jb; label="Jb", lw=1.5, color=:blue, ls=:dash)

    p_ctrl = plot(pVa, pVb, pU, pJ; layout=(2,2), size=(900, 500),
                  plot_title="Controls (N=$N)")

    p_fid = plot(t, fid;
                 xlabel="t", ylabel="F(t)",
                 title="Fidelity vs Time (N=$N, F=$(round(fid[end], digits=4)))",
                 legend=false, lw=1.5, ylim=(0, 1.05), color=:crimson)

    fig = plot(p_ctrl, p_fid; layout=grid(2,1, heights=[0.65, 0.35]), size=(900, 900))
    mkpath("figures")
    out = "figures/exact_sim_N$(N)$(isempty(tag) ? "" : "_" * tag).png"
    savefig(fig, out)
    println("   saved $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Ns = isempty(ARGS) ? [15] : parse.(Int, ARGS)
    for N in Ns
        plot_case_cuda(N)
    end
end
