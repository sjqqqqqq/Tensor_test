using ITensors, ITensorMPS
using JLD2
using Plots

let
    # ── Fixed parameters (match 1d_spin_GRAPE_*.jl) ───────────────────────────
    Jx, Jy, Jz = 2π, 2π, 2π
    T      = 5.0
    nsteps = 500
    dt     = T / nsteps
    cutoff = 1e-10
    maxdim = 100
    @assert Jx ≈ Jy "conserve_qns requires Jx = Jy"

    # Cases to test: N → JLD2 pulse file
    cases = [(11, "1d_spin_11.jld2"),
             (21, "1d_spin_21.jld2"),
             (41, "1d_spin_41.jld2")]

    function run_case(N, pulse_file)
        println("─── N = $N  ($pulse_file) ──────────────────────────────────")
        flush(stdout)

        data = load(pulse_file)
        control = data["control_opt"]
        F_opt_saved = get(data, "F_opt", NaN)
        @assert length(control) == nsteps

        s          = siteinds("S=1/2", N; conserve_qns=true)
        psi        = MPS(s, vcat(["Up"], fill("Dn", N - 1)))
        psi_target = MPS(s, vcat(fill("Dn", N - 1), ["Up"]))

        function heisenberg_bond(k)
            2Jx * op("S+", s[k]) * op("S-", s[k+1]) +
            2Jx * op("S-", s[k]) * op("S+", s[k+1]) +
            4Jz * op("Sz", s[k]) * op("Sz", s[k+1])
        end
        odd_half  = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
        even_half = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
        Sz1 = op("Sz", s[1])

        fidelities = Vector{Float64}(undef, nsteps + 1)
        fidelities[1] = abs2(inner(psi_target, psi))

        t_ev = @elapsed for n in 1:nsteps
            ctrl = exp(-im * dt * 2control[n] * Sz1)
            gates = vcat(odd_half, even_half, [ctrl], even_half, odd_half)
            psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
            normalize!(psi)
            fidelities[n+1] = abs2(inner(psi_target, psi))
        end

        println("  F_opt (saved)   = $(round(F_opt_saved, digits=6))")
        println("  F_check (here)  = $(round(fidelities[end], digits=6))")
        println("  Wall time       = $(round(t_ev, digits=2)) s")
        println()
        flush(stdout)

        # ── Plot: pulse + fidelity vs time ────────────────────────────────────
        t_ctrl = [(n - 0.5) * dt for n in 1:nsteps]       # midpoints
        t_fid  = [(n - 1)   * dt for n in 1:nsteps+1]     # after step n

        p1 = plot(t_ctrl, control;
                  xlabel = "t", ylabel = "c(t)",
                  title  = "Optimized pulse (N=$N)",
                  legend = false, lw = 1.2)
        p2 = plot(t_fid, fidelities;
                  xlabel = "t", ylabel = "F(t)",
                  title  = "Fidelity |⟨ψ_T|ψ(t)⟩|²  (N=$N)",
                  legend = false, lw = 1.2, ylim = (0, 1.05))
        fig = plot(p1, p2; layout = (2, 1), size = (800, 600))
        outfile = "1d_spin_$(N)_test.png"
        savefig(fig, outfile)
        println("  Saved → $outfile\n"); flush(stdout)

        return fidelities, control
    end

    results = Dict{Int,Any}()
    for (N, f) in cases
        results[N] = run_case(N, f)
    end

    # ── Combined comparison plot ──────────────────────────────────────────────
    p_pulse = plot(xlabel = "t", ylabel = "c(t)",
                   title  = "Optimized pulses", lw = 1.2)
    p_fid   = plot(xlabel = "t", ylabel = "F(t)",
                   title  = "Fidelity vs time",
                   ylim   = (0, 1.05), lw = 1.2, legend = :bottomright)
    for (N, _) in cases
        F, c = results[N]
        t_ctrl = [(n - 0.5) * dt for n in 1:nsteps]
        t_fid  = [(n - 1)   * dt for n in 1:nsteps+1]
        plot!(p_pulse, t_ctrl, c; label = "N=$N", lw = 1.0)
        plot!(p_fid,   t_fid,  F; label = "N=$N", lw = 1.2)
    end
    fig_all = plot(p_pulse, p_fid; layout = (2, 1), size = (800, 600))
    savefig(fig_all, "1d_spin_test_all.png")
    println("Saved combined plot → 1d_spin_test_all.png")
end
