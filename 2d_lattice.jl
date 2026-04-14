using ITensors, ITensorMPS

let
    # ── System parameters ─────────────────────────────────────────────────────
    M      = 1        # number of a–b pairs (hard-core: max 1 per species per site)
    T      = 2π       # total time (matches 2d_lattice_GRAPE.jl)
    nsteps = 200      # Trotter steps; dt = T / nsteps
    dt     = T / nsteps
    cutoff = 1e-10
    maxdim = 64

    # ── 4-site ring topology ──────────────────────────────────────────────────
    # Gamma4 bonds (0-indexed): (0,1), (0,2), (1,3), (2,3) — square ring 0–1–3–2–0
    # In 1-indexed ITensor [site0→1, site1→2, site2→3, site3→4]:
    #   Group A (adjacent in MPS):     bonds (1,2) and (3,4)
    #   Group B (non-adjacent in MPS): bonds (1,3) and (2,4)
    bonds_A = [(1,2), (3,4)]
    bonds_B = [(1,3), (2,4)]

    # ── Controls: nsteps × 9  [Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb] ────
    # Va1 = potential difference (site 1 − site 0) for a-type, etc.
    #
    # Test case 1 – constant U/J (non-trivial dynamics):
    controls = zeros(nsteps, 9)
    controls[:, 7] .= 1.0   # U  = 1
    controls[:, 8] .= 1.0   # Ja = 1
    controls[:, 9] .= 1.0   # Jb = 1
    #
    # Test case 2 – zero controls (free evolution; state remains fixed):
    # controls = zeros(nsteps, 9)
    #
    # Test case 3 – sinusoidal Ja/Jb:
    # t_arr = range(0, T; length=nsteps)
    # controls[:, 8] .= 1.0 .+ 0.3 .* sin.(2π .* t_arr ./ T)
    # controls[:, 9] .= 1.0 .+ 0.3 .* cos.(2π .* t_arr ./ T)

    # ── Site indices ──────────────────────────────────────────────────────────
    # "Electron": Up = a-type atom, Dn = b-type atom
    #   Local basis per site: |Emp⟩, |Up⟩, |Dn⟩, |UpDn⟩  (dim = 4)
    #   Hard-core constraint: max 1 a + 1 b per lattice site
    #   QN conservation: total n_up = M, total n_dn = M
    s = siteinds("Electron", 4; conserve_qns=true)

    # ── States ────────────────────────────────────────────────────────────────
    # M=1 initial: a-particle at lattice site 0, b-particle at lattice site 1
    psi0 = MPS(s, ["Up","Dn","Emp","Emp"])

    # M=1 target: (|a@0,b@1⟩ + |a@2,b@3⟩)/√2   (SPDC-like Bell pair)
    psi_t1     = MPS(s, ["Up","Dn","Emp","Emp"])   # |a@0, b@1⟩
    psi_t2     = MPS(s, ["Emp","Emp","Up","Dn"])   # |a@2, b@3⟩
    psi_target = normalize!(add(psi_t1, psi_t2; cutoff, maxdim))

    # M=2 initial (uncomment to use):
    # psi0 = MPS(s, ["Up","Up","Dn","Dn"])   # a@{0,1}, b@{2,3}
    # M=2 target: (|a@{0,1},b@{2,3}⟩ + |a@{2,3},b@{0,1}⟩)/√2
    # psi_target = normalize!(add(MPS(s,["Up","Up","Dn","Dn"]),
    #                             MPS(s,["Dn","Dn","Up","Up"]); cutoff, maxdim))

    println("System  : 4-site ring (Gamma4), M=$M a-b pair(s), hard-core bosons")
    println("Time    : T=$(round(T,digits=4)), nsteps=$nsteps, dt=$(round(dt,digits=5))")
    println("Initial : |a@0, b@1⟩")
    println("Target  : (|a@0,b@1⟩ + |a@2,b@3⟩)/√2")
    println()
    println("Initial state norm    : ", norm(psi0))
    println("Initial fidelity F(0) : ", abs2(inner(psi_target, psi0)))
    println()

    # ── On-site Hamiltonian coefficients ──────────────────────────────────────
    # H_onsite = Va1(n^a_1 − n^a_0) + Va2(n^a_2 − n^a_0) + Va3(n^a_3 − n^a_0)
    #          + Vb1(n^b_1 − n^b_0) + Vb2(n^b_2 − n^b_0) + Vb3(n^b_3 − n^b_0)
    #          + U·Σ_j n^a_j n^b_j
    # Per-site coefficient for a-type (j is 1-indexed ITensor site):
    va_coeff(j, Va1,Va2,Va3) = j==1 ? -(Va1+Va2+Va3) :
                                j==2 ?   Va1 :
                                j==3 ?   Va2 : Va3
    vb_coeff(j, Vb1,Vb2,Vb3) = j==1 ? -(Vb1+Vb2+Vb3) :
                                 j==2 ?   Vb1 :
                                 j==3 ?   Vb2 : Vb3

    function make_onsite_gates(Va1,Va2,Va3, Vb1,Vb2,Vb3, U, τ)
        [exp(-im*τ*(va_coeff(j,Va1,Va2,Va3)*op("Nup",  s[j]) +
                    vb_coeff(j,Vb1,Vb2,Vb3)*op("Ndn",  s[j]) +
                    U                       *op("Nupdn",s[j]))) for j in 1:4]
    end

    # ── Hopping gates (treated as hard-core bosons; no Jordan-Wigner sign) ───
    make_hop_a(j,k,Ja,τ) = exp(-im*τ*Ja*(op("Cdagup",s[j])*op("Cup",   s[k]) +
                                          op("Cup",   s[j])*op("Cdagup",s[k])))
    make_hop_b(j,k,Jb,τ) = exp(-im*τ*Jb*(op("Cdagdn",s[j])*op("Cdn",   s[k]) +
                                          op("Cdn",   s[j])*op("Cdagdn",s[k])))

    # ── Single 2nd-order Trotter step ─────────────────────────────────────────
    #
    # Structure: D1_half · U_Ja · U_Jb · D1_half
    #   D1_half = exp(−i·dt/2·H_onsite)      [4 independent single-site gates]
    #   U_Ja    = exp(−i·Ja·dt·H_Ja)         [2nd-order sub-Trotter: A/2→B→A/2]
    #   U_Jb    = exp(−i·Jb·dt·H_Jb)         [same structure]
    #
    # H_Ja = Σ_{bonds} (a†_j a_k + h.c.),  split into:
    #   Group A: bonds (1,2),(3,4) — adjacent in MPS, commute with each other
    #   Group B: bonds (1,3),(2,4) — non-adjacent, handled by ITensor SWAP
    # [H_Ja_A, H_Ja_B] ≠ 0 → 2nd-order sub-Trotter error O(dt³·Ja²) per step.
    function trotter_step(psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, step_dt)
        d1h   = make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, step_dt/2)
        jaAh  = [make_hop_a(j,k,Ja,step_dt/2) for (j,k) in bonds_A]
        jaBf  = [make_hop_a(j,k,Ja,step_dt)   for (j,k) in bonds_B]
        jbAh  = [make_hop_b(j,k,Jb,step_dt/2) for (j,k) in bonds_A]
        jbBf  = [make_hop_b(j,k,Jb,step_dt)   for (j,k) in bonds_B]

        # Apply all gates in a single sweep: D1/2 · jaA/2 · jaB · jaA/2 · jbA/2 · jbB · jbA/2 · D1/2
        psi = apply(vcat(d1h, jaAh, jaBf, jaAh, jbAh, jbBf, jbAh, d1h), psi; cutoff, maxdim)
        normalize!(psi)
        return psi
    end

    # ── JIT warmup ────────────────────────────────────────────────────────────
    println("JIT warmup..."); flush(stdout)
    psi_warm = copy(psi0)
    for n in 1:3
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[n,:]
        psi_warm = trotter_step(psi_warm, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
        _ = abs2(inner(psi_target, psi_warm))
        _ = real(expect(psi_warm, "Nup"))
    end
    println("JIT warmup done.\n"); flush(stdout)

    # ── Observables storage ───────────────────────────────────────────────────
    fidelities   = Vector{Float64}(undef, nsteps+1)
    nup_profiles = Matrix{Float64}(undef, nsteps+1, 4)
    ndn_profiles = Matrix{Float64}(undef, nsteps+1, 4)
    bond_dims    = Vector{Int}(undef,    nsteps+1)

    fidelities[1]       = abs2(inner(psi_target, psi0))
    nup_profiles[1, :] .= real(expect(psi0, "Nup"))
    ndn_profiles[1, :] .= real(expect(psi0, "Ndn"))
    bond_dims[1]        = maxlinkdim(psi0)

    # ── Time evolution loop ───────────────────────────────────────────────────
    println("Starting time evolution: M=$M, T=$(round(T,digits=4)), nsteps=$nsteps, dt=$(round(dt,digits=5))")
    flush(stdout)

    psi = copy(psi0)
    t_evolve = @elapsed for n in 1:nsteps
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[n,:]
        psi = trotter_step(psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
        fidelities[n+1]       = abs2(inner(psi_target, psi))
        nup_profiles[n+1, :] .= real(expect(psi, "Nup"))
        ndn_profiles[n+1, :] .= real(expect(psi, "Ndn"))
        bond_dims[n+1]        = maxlinkdim(psi)
    end

    # ── Print summary ─────────────────────────────────────────────────────────
    println("─── Time evolution complete ───────────────────────────────────────")
    println("Wall time (evolution loop) : $(round(t_evolve, digits=2)) s  ($(round(t_evolve/nsteps*1000, digits=3)) ms/step)")
    println("Final fidelity F(T)        : $(fidelities[end])")
    println("Max bond dim at t=T        : $(bond_dims[end])")
    println("Max bond dim overall       : $(maximum(bond_dims))")
    println()
    println("n_up profile at t=0 : ", round.(nup_profiles[1,:],   digits=3))
    println("n_up profile at t=T : ", round.(nup_profiles[end,:], digits=3))
    println("n_dn profile at t=0 : ", round.(ndn_profiles[1,:],   digits=3))
    println("n_dn profile at t=T : ", round.(ndn_profiles[end,:], digits=3))
    println()

    # Fidelity at selected time points
    println("Fidelity evolution (selected steps):")
    checkpoints = unique([1; 20:20:nsteps; nsteps+1])
    for n in checkpoints
        t = (n-1) * dt
        println("  t = $(lpad(round(t,digits=3),6))   " *
                "F = $(round(fidelities[n],digits=6))   " *
                "χ_max = $(bond_dims[n])")
    end

    # ── Trotter accuracy check (reference: dt_ref = dt/10) ────────────────────
    # 2nd-order Trotter has global error O(dt²); expect |ΔF| ≈ 100× smaller at dt/10.
    println()
    println("─── Trotter accuracy check (reference dt = $(round(dt/10,digits=6))) ──────────────")
    dt_ref  = dt / 10
    ns_ref  = nsteps * 10
    Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[1,:]   # constant controls assumed
    # Precompute reference gates once (constant controls)
    d1h_ref  = make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt_ref/2)
    jaAh_ref = [make_hop_a(j,k,Ja,dt_ref/2) for (j,k) in bonds_A]
    jaBf_ref = [make_hop_a(j,k,Ja,dt_ref)   for (j,k) in bonds_B]
    jbAh_ref = [make_hop_b(j,k,Jb,dt_ref/2) for (j,k) in bonds_A]
    jbBf_ref = [make_hop_b(j,k,Jb,dt_ref)   for (j,k) in bonds_B]
    ref_gates = vcat(d1h_ref, jaAh_ref, jaBf_ref, jaAh_ref, jbAh_ref, jbBf_ref, jbAh_ref, d1h_ref)

    psi_ref = copy(psi0)
    t_ref = @elapsed for _ in 1:ns_ref
        psi_ref = apply(ref_gates, psi_ref; cutoff, maxdim)
        normalize!(psi_ref)
    end
    F_ref = abs2(inner(psi_target, psi_ref))
    println("Reference fidelity (dt=$(round(dt_ref,digits=6)), $ns_ref steps) : $(round(F_ref, digits=8))")
    println("Coarse    fidelity (dt=$(round(dt,digits=5)),  $nsteps steps) : $(round(fidelities[end], digits=8))")
    println("Trotter error |ΔF|                               : $(round(abs(fidelities[end] - F_ref), sigdigits=3))")
    println("Reference wall time                              : $(round(t_ref, digits=2)) s  ($(round(t_ref/ns_ref*1000, digits=3)) ms/step)")

    (fidelities=fidelities, nup_profiles=nup_profiles, ndn_profiles=ndn_profiles, bond_dims=bond_dims)
end
