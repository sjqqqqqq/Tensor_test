using ITensors, ITensorMPS

let
    # ── System parameters ─────────────────────────────────────────────────────
    N  = 11          # number of spins (d = 11 from the paper)
    Jx = 2π          # XXX model: Jx = Jy = Jz = 2π  (paper Section 4 / Fig. 1)
    Jy = 2π          # Set Jz = 2.2π here for XXZ model
    Jz = 2π

    # ── Time-evolution parameters ─────────────────────────────────────────────
    T      = 5.0     # total evolution time (paper uses T = 5 for d = 11)
    nsteps = 500     # number of Trotter steps  →  dt = 0.01
    dt     = T / nsteps
    cutoff = 1e-10   # SVD truncation cutoff
    maxdim = 100     # max MPS bond dimension

    # ── Control input (piecewise-constant, one value per Trotter step) ────────
    # Each entry c[n] is the amplitude of the control field H_c(t) = c(t)·σz^(1)
    # on the interval [(n-1)·dt, n·dt].
    #
    # Test case 1 – zero control: free Heisenberg evolution
    control = zeros(nsteps)

    # Test case 2 – constant control (uncomment to use):
    # control = fill(1.0, nsteps)

    # Test case 3 – sinusoidal control (uncomment to use):
    # control = [2π * sin(2π * (n - 0.5) * dt) for n in 1:nsteps]

    # ── Site indices ──────────────────────────────────────────────────────────
    # conserve_qns=true exploits total-Sz conservation (preserved by both the
    # Heisenberg drift and the σz control) for a significant speed-up.
    s = siteinds("S=1/2", N; conserve_qns=true)

    # ── Initial state |ψ₀⟩ = |↑↓↓···↓↓⟩ ─────────────────────────────────────
    # Paper: "up" excitation on site 1, all other sites spin-down.
    psi = MPS(s, vcat(["Up"], fill("Dn", N - 1)))

    # ── Target state |ψ_T⟩ = |↓↓···↓↓↑⟩ ─────────────────────────────────────
    # Paper: "up" excitation transferred to last site.
    psi_target = MPS(s, vcat(fill("Dn", N - 1), ["Up"]))

    println("Initial state norm      : ", norm(psi))
    println("Initial fidelity F(0)   : ", abs2(inner(psi_target, psi)))
    println()

    # ── Two-site Heisenberg bond Hamiltonian ──────────────────────────────────
    # Paper Hamiltonian (Eq. 11) uses Pauli matrices σ (eigenvalues ±1).
    # ITensor "S=1/2": Sα = σα/2, S± = σ±.
    #
    # "Sy" alone is not block-diagonal in total-Sz and cannot be used with
    # conserve_qns=true.  For Jx = Jy (XXX / XXZ in the paper) we use:
    #   σx_k⊗σx_{k+1} + σy_k⊗σy_{k+1} = 2(S+_k⊗S-_{k+1} + S-_k⊗S+_{k+1})
    # which is QN-conserving (net ΔSz = 0).
    # z-term:  σz_k⊗σz_{k+1} = 4 Sz_k⊗Sz_{k+1}.
    @assert Jx ≈ Jy "conserve_qns requires Jx = Jy (XXX or XXZ). Set conserve_qns=false for XYZ."
    function heisenberg_bond(k)
        return 2Jx * op("S+", s[k]) * op("S-", s[k+1]) +
               2Jx * op("S-", s[k]) * op("S+", s[k+1]) +
               4Jz * op("Sz", s[k]) * op("Sz", s[k+1])
    end

    # ── Pre-compute half-step drift gates (computed once; reused every step) ──
    #
    # 2nd-order Strang-Trotter splitting for H = H_odd + H_even + H_ctrl:
    #
    #   U(dt) ≈  ∏_{k odd}  e^{-i·dt/2·h_k}
    #          · ∏_{k even} e^{-i·dt/2·h_k}
    #          ·             e^{-i·dt·H_ctrl}        ← rebuilt each step
    #          · ∏_{k even} e^{-i·dt/2·h_k}
    #          · ∏_{k odd}  e^{-i·dt/2·h_k}
    #
    # Operators on non-overlapping bonds commute, so order within each group
    # is arbitrary.
    t_gate = @elapsed begin
        odd_half  = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
        even_half = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    end
    println("Pre-computed $(length(odd_half)) odd + $(length(even_half)) even half-step drift gates  ($( round(t_gate, digits=2)) s).")
    println("Starting time evolution : N=$N spins, T=$T, nsteps=$nsteps, dt=$dt")
    println()

    # ── Observables storage ───────────────────────────────────────────────────
    fidelities  = Vector{Float64}(undef, nsteps + 1)
    sz_profiles = Matrix{Float64}(undef, nsteps + 1, N)
    bond_dims   = Vector{Int}(undef, nsteps + 1)

    # Record initial state
    fidelities[1]      = abs2(inner(psi_target, psi))
    sz_profiles[1, :] .= real(expect(psi, "Sz"))
    bond_dims[1]       = maxlinkdim(psi)

    # ── JIT warmup (10 steps; results discarded) ─────────────────────────────
    psi_warm = copy(psi)
    for n in 1:10
        ctrl_gate = exp(-im * dt * 0.0 * op("Sz", s[1]))
        gates_w   = vcat(odd_half, even_half, [ctrl_gate], even_half, odd_half)
        psi_warm  = apply(gates_w, psi_warm; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi_warm)
        _ = abs2(inner(psi_target, psi_warm))
        _ = real(expect(psi_warm, "Sz"))
    end
    println("JIT warmup done.")

    # ── Time evolution loop ───────────────────────────────────────────────────
    t_evolve = @elapsed for n in 1:nsteps
        c_n = control[n]

        # Single-site control gate (full step):
        #   H_ctrl = c(t)·σz^(1)  =  2·c(t)·Sz^(1)   [Pauli → ITensor convention]
        ctrl_gate = exp(-im * dt * 2c_n * op("Sz", s[1]))

        # Assemble and apply the full 2nd-order Trotter gate sequence:
        # odd/2 | even/2 | ctrl | even/2 | odd/2
        gates = vcat(odd_half, even_half, [ctrl_gate], even_half, odd_half)
        psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi)

        # Record observables after this step
        fidelities[n+1]       = abs2(inner(psi_target, psi))
        sz_profiles[n+1, :] .= real(expect(psi, "Sz"))
        bond_dims[n+1]        = maxlinkdim(psi)
    end

    # ── Print summary ─────────────────────────────────────────────────────────
    println("─── Time evolution complete ─────────────────────────────────────")
    println("Wall time (evolution loop) : $(round(t_evolve, digits=2)) s  ($(round(t_evolve/nsteps*1000, digits=3)) ms/step)")
    println("Final fidelity |⟨ψ_T|ψ(T)⟩|²  = ", fidelities[end])
    println("Max bond dimension at t=T       = ", bond_dims[end])
    println("Max bond dimension overall      = ", maximum(bond_dims))
    println()
    println("Sz profile at t=0 : ", round.(sz_profiles[1,   :], digits=3))
    println("Sz profile at t=T : ", round.(sz_profiles[end, :], digits=3))
    println()

    # Fidelity at selected time points
    println("Fidelity evolution (selected steps):")
    checkpoints = unique([1; 50:50:nsteps; nsteps+1])
    for n in checkpoints
        t = (n - 1) * dt
        println("  t = $(lpad(round(t, digits=2), 5))   " *
                "F = $(round(fidelities[n], digits=6))   " *
                "χ_max = $(bond_dims[n])")
    end

    # ── Trotter accuracy test: compare dt=0.01 vs dt=0.001 ───────────────────
    # A 2nd-order Trotter scheme has global error O(dt²).
    # Running 10× finer (dt_ref = dt/10) gives a reference trajectory.
    # The difference in final fidelity estimates the Trotter error.
    println("─── Trotter accuracy check (reference: dt=$(dt/10)) ──────────────")
    dt_ref   = dt / 10
    ns_ref   = nsteps * 10
    oh_ref   = [exp(-im * dt_ref/2 * heisenberg_bond(k)) for k in 1:2:N-1]
    eh_ref   = [exp(-im * dt_ref/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    psi_ref  = MPS(s, vcat(["Up"], fill("Dn", N - 1)))  # fresh initial state
    ctrl_ref = exp(-im * dt_ref * 0.0 * op("Sz", s[1])) # zero control
    t_ref = @elapsed for _ in 1:ns_ref
        gates_r = vcat(oh_ref, eh_ref, [ctrl_ref], eh_ref, oh_ref)
        psi_ref = apply(gates_r, psi_ref; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi_ref)
    end
    F_ref = abs2(inner(psi_target, psi_ref))
    println("Reference fidelity (dt=$(dt/10), $ns_ref steps) : $(round(F_ref, digits=8))")
    println("Coarse    fidelity (dt=$dt, $nsteps steps)  : $(round(fidelities[end], digits=8))")
    println("Trotter error |ΔF|                          : $(round(abs(fidelities[end] - F_ref), sigdigits=3))")
    println("Reference wall time                         : $(round(t_ref, digits=2)) s  ($(round(t_ref/ns_ref*1000, digits=3)) ms/step)")

    # Return observables for interactive use
    (fidelities=fidelities, sz_profiles=sz_profiles, bond_dims=bond_dims)
end
