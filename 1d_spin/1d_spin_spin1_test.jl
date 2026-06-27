using ITensors, ITensorMPS
using JLD2
using Plots

# Replay the best multistart pulse for the spin-1-first-site chain and plot the
# optimal control c(t) together with the fidelity F(t), matching the layout of
# 1d_spin_11_test.png (pulse on top, fidelity below).

let
    # ── Fixed parameters (match 1d_spin_GRAPE_spin1_multistart.jl) ────────────
    Jx, Jy, Jz = 2π, 2π, 2π
    T      = 15.0
    nsteps = 1500
    dt     = T / nsteps
    cutoff = 1e-10
    maxdim = 100
    @assert Jx ≈ Jy "conserve_qns requires Jx = Jy"

    pulse_file = joinpath(@__DIR__, "1d_spin_spin1.jld2")
    data    = load(pulse_file)
    control = data["control_opt"]
    F_saved = get(data, "F_opt", NaN)
    N       = get(data, "N", 11)
    @assert length(control) == nsteps "pulse length $(length(control)) != nsteps $nsteps"

    # ── Site indices, states (spin-1 on site 1, spin-1/2 elsewhere) ──────────
    s          = siteinds(n -> n == 1 ? "S=1" : "S=1/2", N; conserve_qns=true)
    psi        = MPS(s, vcat(["Up"], fill("Dn", N - 1)))            # |m=+1,↓···↓⟩
    psi_target = MPS(s, vcat(["Z0"], fill("Dn", N - 2), ["Up"]))   # |m=0,↓···↓↑⟩

    function heisenberg_bond(k)
        (Jx/2) * op("S+", s[k]) * op("S-", s[k+1]) +
        (Jx/2) * op("S-", s[k]) * op("S+", s[k+1]) +
        Jz     * op("Sz", s[k]) * op("Sz", s[k+1])
    end
    odd_half  = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
    even_half = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    Sz1 = op("Sz", s[1])

    fidelities = Vector{Float64}(undef, nsteps + 1)
    fidelities[1] = abs2(inner(psi_target, psi))

    t_ev = @elapsed for n in 1:nsteps
        ctrl  = exp(-im * dt * control[n] * Sz1)
        gates = vcat(odd_half, even_half, [ctrl], even_half, odd_half)
        psi   = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi)
        fidelities[n+1] = abs2(inner(psi_target, psi))
    end

    println("─── N = $N  ($(basename(pulse_file))) ───")
    println("  F_opt (saved)   = $(round(F_saved, digits=6))")
    println("  F_check (here)  = $(round(fidelities[end], digits=6))")
    println("  Wall time       = $(round(t_ev, digits=2)) s")
    flush(stdout)

    # ── Plot: pulse + fidelity vs time ────────────────────────────────────────
    t_ctrl = [(n - 0.5) * dt for n in 1:nsteps]       # control midpoints
    t_fid  = [(n - 1)   * dt for n in 1:nsteps+1]     # after step n

    p1 = plot(t_ctrl, control;
              xlabel = "t", ylabel = "c(t)",
              title  = "Optimized pulse (N=$N, S=1 site 1)",
              legend = false, lw = 1.2)
    p2 = plot(t_fid, fidelities;
              xlabel = "t", ylabel = "F(t)",
              title  = "Fidelity |⟨ψ_T|ψ(t)⟩|²  (F=$(round(fidelities[end], digits=5)))",
              legend = false, lw = 1.2, ylim = (0, 1.05))
    fig = plot(p1, p2; layout = (2, 1), size = (800, 600))
    outfile = joinpath(@__DIR__, "1d_spin_spin1_test.png")
    savefig(fig, outfile)
    println("  Saved → $outfile"); flush(stdout)
end
