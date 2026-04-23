using JLD2
using Plots

include("soft_boson.jl")

function trotter_step(s, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, step_dt;
                      cutoff, maxdim)
    # 4 apply+normalize sub-steps — matches MPS_GRAPE.jl's forward pass so the
    # simulated F(T) is bit-identical to the fidelity reported by GRAPE.
    d1h = make_onsite_gates(s, Va1,Va2,Va3, Vb1,Vb2,Vb3, U, step_dt/2)
    psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi)
    psi = apply(hop_gates_a(s, Ja, step_dt), psi; cutoff, maxdim); normalize!(psi)
    psi = apply(hop_gates_b(s, Jb, step_dt), psi; cutoff, maxdim); normalize!(psi)
    psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi)
    return psi
end

function simulate_case(N::Int; cutoff=1e-10, maxdim=64,
                       file::String = "data/GRAPE_2d_$(N).jld2",
                       tag::String = "")
    println("="^70)
    println("MPS simulation — N = $N pair(s)")
    println("="^70)

    pulse_file = file
    println("Loading GRAPE pulse from: $pulse_file")
    pulse   = load(pulse_file)
    n_pulse = Int(pulse["n"]); nsteps = n_pulse - 1
    dt      = pulse["dt"]; T = pulse["T"]
    Va1_p   = pulse["Va1"][1:nsteps]; Va2_p = pulse["Va2"][1:nsteps]; Va3_p = pulse["Va3"][1:nsteps]
    Vb1_p   = pulse["Vb1"][1:nsteps]; Vb2_p = pulse["Vb2"][1:nsteps]; Vb3_p = pulse["Vb3"][1:nsteps]
    U_p     = pulse["U"][1:nsteps];   Ja_p  = pulse["Ja"][1:nsteps];  Jb_p  = pulse["Jb"][1:nsteps]
    controls = hcat(Va1_p, Va2_p, Va3_p, Vb1_p, Vb2_p, Vb3_p, U_p, Ja_p, Jb_p)
    grape_fidelity = pulse["fidelity"]
    println("  n_pulse=$n_pulse, T=$(round(T,digits=4)), dt=$(round(dt,digits=6))")
    println("  GRAPE (exact) fidelity: $(round(grape_fidelity, digits=8))")

    s = siteinds("SoftBoson", 4)
    psi0, psi_target = default_states(s, N; cutoff, maxdim)

    println("Initial state norm    : ", norm(psi0))
    println("Initial fidelity F(0) : ", abs2(inner(psi_target, psi0)))

    # JIT warmup
    psi_warm = copy(psi0)
    for n in 1:min(3, nsteps)
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[n,:]
        psi_warm = trotter_step(s, psi_warm, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt;
                                cutoff, maxdim)
    end

    fidelities = Vector{Float64}(undef, nsteps+1)
    bond_dims  = Vector{Int}(undef,     nsteps+1)
    fidelities[1] = abs2(inner(psi_target, psi0))
    bond_dims[1]  = maxlinkdim(psi0)

    psi = copy(psi0)
    t_evolve = @elapsed for n in 1:nsteps
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[n,:]
        psi = trotter_step(s, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt;
                           cutoff, maxdim)
        fidelities[n+1] = abs2(inner(psi_target, psi))
        bond_dims[n+1]  = maxlinkdim(psi)
    end

    println("─── Time evolution complete ───")
    println("Wall time : $(round(t_evolve, digits=2)) s  ($(round(t_evolve/nsteps*1000, digits=3)) ms/step)")
    println("Final F(T): $(fidelities[end])")
    println("Max χ     : $(maximum(bond_dims))")
    println("GRAPE F   : $(round(grape_fidelity, digits=8))")
    println("|ΔF|      : $(round(abs(fidelities[end] - grape_fidelity), sigdigits=3))")

    # Plots
    t_grid = (0:nsteps) .* dt
    ctrl_names = ["Va1","Va2","Va3","Vb1","Vb2","Vb3","U","Ja","Jb"]
    ctrl_plots = [plot(t_grid[1:nsteps], controls[:,k]; title=ctrl_names[k],
                       xlabel="t", ylabel=ctrl_names[k], legend=false, lw=1.2)
                  for k in 1:9]
    p_ctrl = plot(ctrl_plots..., layout=(3,3), size=(900,700),
                  plot_title="MPS sim controls (N=$N)")
    p_fid = plot(t_grid, fidelities;
                 xlabel="t", ylabel="F(t)",
                 title="MPS Fidelity (N=$N, F(T)=$(round(fidelities[end], digits=4)))",
                 legend=false, lw=1.5, ylim=(0, 1.05), color=:crimson)
    fig = plot(p_ctrl, p_fid; layout=grid(2,1, heights=[0.65, 0.35]), size=(900,900))
    out = "figures/MPS_sim_N$(N)$(isempty(tag) ? "" : "_" * tag).png"
    savefig(fig, out)
    println("  Saved $out")
    return (fidelities=fidelities, bond_dims=bond_dims, grape_fidelity=grape_fidelity)
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = joinpath(@__DIR__, "data")
    for N in 1:3
        # Exact-GRAPE pulse
        ef = joinpath(data_dir, "GRAPE_2d_$(N).jld2")
        isfile(ef) && simulate_case(N; file=ef, tag="exact")
        # MPS-GRAPE (Trotter) pulse
        mf = joinpath(data_dir, "GRAPE_2d_MPS_$(N).jld2")
        isfile(mf) && simulate_case(N; file=mf, tag="MPS")
        # MPS-GRAPE (MPO-TDVP) pulse — cross-check via Trotter replay
        tf = joinpath(data_dir, "GRAPE_2d_MPS_TDVP_$(N).jld2")
        isfile(tf) && simulate_case(N; file=tf, tag="MPS_TDVP")
    end
end
