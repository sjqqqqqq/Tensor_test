using ITensors, ITensorMPS
using JLD2
using Plots

let
    Jx, Jy, Jz = 2π, 2π, 2π
    T      = 10.0
    nsteps = 500
    dt     = T / nsteps       # = 0.02
    cutoff = 1e-10
    maxdim = 100
    N      = 41

    data        = load("1d_spin_41.jld2")
    control     = data["control_opt"]
    F_opt_saved = get(data, "F_opt", NaN)
    @assert length(control) == nsteps

    s          = siteinds("S=1/2", N; conserve_qns=true)
    psi        = MPS(s, vcat(["Up"], fill("Dn", N - 1)))
    psi_target = MPS(s, vcat(fill("Dn", N - 1), ["Up"]))

    function hb(k)
        2Jx*op("S+",s[k])*op("S-",s[k+1]) +
        2Jx*op("S-",s[k])*op("S+",s[k+1]) +
        4Jz*op("Sz",s[k])*op("Sz",s[k+1])
    end
    odd_half  = [exp(-im*dt/2*hb(k)) for k in 1:2:N-1]
    even_half = [exp(-im*dt/2*hb(k)) for k in 2:2:N-1]
    Sz1 = op("Sz", s[1])

    fidelities = Vector{Float64}(undef, nsteps + 1)
    fidelities[1] = abs2(inner(psi_target, psi))

    t_ev = @elapsed for n in 1:nsteps
        ctrl  = exp(-im*dt*2control[n]*Sz1)
        gates = vcat(odd_half, even_half, [ctrl], even_half, odd_half)
        psi   = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi)
        fidelities[n+1] = abs2(inner(psi_target, psi))
    end

    println("N = $N, T = $T, dt = $dt")
    println("  F_opt (saved, T=5 run) = ", round(F_opt_saved, digits=6))
    println("  F(T=$T)                = ", round(fidelities[end], digits=6))
    println("  Wall time              = ", round(t_ev, digits=2), " s")

    t_ctrl = [(n - 0.5)*dt for n in 1:nsteps]
    t_fid  = [(n - 1)  *dt for n in 1:nsteps+1]

    p1 = plot(t_ctrl, control;
              xlabel="t", ylabel="c(t)",
              title="Optimized pulse (N=$N, T=$T, dt=$dt)",
              legend=false, lw=1.2)
    p2 = plot(t_fid, fidelities;
              xlabel="t", ylabel="F(t)",
              title="Fidelity |⟨ψ_T|ψ(t)⟩|²  (N=$N, T=$T)",
              legend=false, lw=1.2, ylim=(0, 1.05))
    fig = plot(p1, p2; layout=(2,1), size=(800,600))
    savefig(fig, "1d_spin_41_test.png")
    println("Saved → 1d_spin_41_test.png")
end
