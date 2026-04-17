using Pkg
Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using LinearAlgebra, Random, Statistics
using Optim
using Plots

let
    # ── System parameters ────────────────────────────────────────────────────
    N      = 11           # spins (d = 11, paper Section 4)
    Jx     = 2π           # XXX Heisenberg: Jx = Jy = Jz = 2π
    Jy     = 2π           # (set Jz = 2.2π for XXZ)
    Jz     = 2π
    T      = 5.0          # total time (paper uses T = 5 for d = 11)
    nsteps = 500          # Trotter steps, dt = 0.01 (paper: τ = 10⁻²)
    dt     = T / nsteps
    cutoff = 1e-10
    maxdim = 100

    @assert Jx ≈ Jy "conserve_qns requires Jx = Jy (XXX or XXZ)"

    # ── Site indices, states ──────────────────────────────────────────────────
    s          = siteinds("S=1/2", N; conserve_qns=true)
    psi0       = MPS(s, vcat(["Up"], fill("Dn", N - 1)))    # |↑↓↓···↓↓⟩
    psi_target = MPS(s, vcat(fill("Dn", N - 1), ["Up"]))    # |↓↓···↓↓↑⟩

    # ── Drift half-step gates (time-independent; computed once) ───────────────
    # Bond: Jx·σx_k⊗σx_{k+1} + Jz·σz_k⊗σz_{k+1}
    # QN-safe form for Jx = Jy:  2Jx·(S+S- + S-S+) + 4Jz·Sz⊗Sz
    function heisenberg_bond(k)
        2Jx * op("S+", s[k]) * op("S-", s[k+1]) +
        2Jx * op("S-", s[k]) * op("S+", s[k+1]) +
        4Jz * op("Sz", s[k]) * op("Sz", s[k+1])
    end

    println("Pre-computing drift gates..."); flush(stdout)
    odd_half      = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
    even_half     = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    adj_odd_half  = [exp(+im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
    adj_even_half = [exp(+im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    Sz1 = op("Sz", s[1])     # single-site operator for gradient
    println("Done ($(length(odd_half)) odd + $(length(even_half)) even bonds).\n"); flush(stdout)

    # ── Trotter step structure ────────────────────────────────────────────────
    #
    # Forward step n:   |ψ_n⟩ = P₂ · G_ctrl_n · P₁ · |ψ_{n-1}⟩
    #   P₁  = P_even · P_odd   ← apply(vcat(odd_half,  even_half), ψ)
    #   P₂  = P_odd  · P_even  ← apply(vcat(even_half, odd_half),  ψ)
    #   G_ctrl_n = exp(-im·dt·c_n·σz^(1)) = exp(-im·dt·2c_n·Sz₁)
    #
    # Backward step n:  χ^{n-1} = P₁† · G_ctrl_n† · P₂† · χ^n
    #   P₂† ← apply(vcat(adj_odd_half,  adj_even_half), χ)
    #   P₁† ← apply(vcat(adj_even_half, adj_odd_half),  χ)
    #
    # Gradient formula (exact Trotter-GRAPE, one control site σz^(1)):
    #   ∂F/∂c_n = 2·Re[ overlap* · (-im·dt) · ⟨P₂†χ^n | σz^(1) | φ_n⟩ ]
    #           = 2·Re[ conj(overlap) · (-im·dt) · 2·⟨λ_bc | Sz₁ | φ_n⟩ ]
    # where φ_n = G_ctrl_n · P₁ · ψ_{n-1}  (state AFTER ctrl gate, stored below).

    # ── Forward pass: stores |φ_n⟩ for gradient computation ──────────────────
    function forward(control)
        psi      = copy(psi0)
        post_ctrl = Vector{MPS}(undef, nsteps)
        for n in 1:nsteps
            # First half of drift: P₁ = P_even · P_odd
            psi = apply(vcat(odd_half, even_half), psi; cutoff, maxdim)
            normalize!(psi)
            # Control gate: exp(-im·dt·c_n·σz^(1)) = exp(-im·dt·2c_n·Sz₁)
            ctrl = exp(-im * dt * 2control[n] * Sz1)
            psi = apply([ctrl], psi; cutoff, maxdim)
            normalize!(psi)
            post_ctrl[n] = copy(psi)     # |φ_n⟩ = state after ctrl gate
            # Second half of drift: P₂ = P_odd · P_even
            psi = apply(vcat(even_half, odd_half), psi; cutoff, maxdim)
            normalize!(psi)
        end
        return psi, post_ctrl
    end

    # ── Fidelity only (no state storage; used for verification) ──────────────
    function fidelity_only(control)
        psi = copy(psi0)
        for n in 1:nsteps
            ctrl = exp(-im * dt * 2control[n] * Sz1)
            psi  = apply(vcat(odd_half, even_half, [ctrl], even_half, odd_half),
                         psi; cutoff, maxdim)
            normalize!(psi)
        end
        return abs2(inner(psi_target, psi))
    end

    # ── Infidelity vs time (for post-optimization diagnostics) ──────────────
    function infidelity_vs_time(control)
        psi = copy(psi0)
        infid = zeros(Float64, nsteps)
        for n in 1:nsteps
            ctrl = exp(-im * dt * 2control[n] * Sz1)
            psi  = apply(vcat(odd_half, even_half, [ctrl], even_half, odd_half),
                         psi; cutoff, maxdim)
            normalize!(psi)
            infid[n] = 1.0 - abs2(inner(psi_target, psi))
        end
        return infid
    end

    # ── Fidelity + gradient (combined forward-backward GRAPE sweep) ───────────
    function compute_fg(control)
        # ── Forward pass ──────────────────────────────────────────────────────
        psi_final, post_ctrl = forward(control)
        overlap = inner(psi_target, psi_final)
        F       = abs2(overlap)

        # ── Backward pass + gradient ──────────────────────────────────────────
        chi  = copy(psi_target)
        grad = zeros(Float64, nsteps)

        for n in nsteps:-1:1
            c_n = control[n]

            # Step 1: λ_bc = P₂† · χ^n  (costate at position of |φ_n⟩)
            chi = apply(vcat(adj_odd_half, adj_even_half), chi; cutoff, maxdim)
            normalize!(chi)

            # Step 2: ∂F/∂c_n = 2·Re[overlap* · (-im·dt) · 2·⟨λ_bc|Sz₁|φ_n⟩]
            psi_tmp    = copy(post_ctrl[n])
            psi_tmp[1] = noprime(Sz1 * psi_tmp[1])   # apply Sz to site 1
            grad[n]    = 2real(conj(overlap) * (-im * dt) * 2inner(chi, psi_tmp))

            # Step 3: G_ctrl_n† · λ_bc
            chi = apply([exp(+im * dt * 2c_n * Sz1)], chi; cutoff, maxdim)
            normalize!(chi)

            # Step 4: χ^{n-1} = P₁† · G_ctrl_n† · λ_bc
            chi = apply(vcat(adj_even_half, adj_odd_half), chi; cutoff, maxdim)
            normalize!(chi)
        end

        return F, grad
    end

    # ── JIT warmup ────────────────────────────────────────────────────────────
    println("JIT warmup..."); flush(stdout)
    compute_fg(zeros(nsteps))
    fidelity_only(zeros(nsteps))
    infidelity_vs_time(zeros(nsteps))
    println("Warmup done.\n"); flush(stdout)

    # ── Gradient accuracy check (analytic vs finite-difference) ──────────────
    println("Gradient check (analytic vs central-difference, 5 components:)"); flush(stdout)
    c_test = 0.3 * randn(MersenneTwister(0), nsteps)
    F0, ag = compute_fg(c_test)
    h      = 1e-5
    println("  Reference fidelity F = $(round(F0, digits=6))"); flush(stdout)
    for i in [1, 50, 100, 250, 499]
        cp_p = copy(c_test); cp_p[i] += h
        cp_m = copy(c_test); cp_m[i] -= h
        ng_i = (fidelity_only(cp_p) - fidelity_only(cp_m)) / (2h)
        err  = abs(ag[i] - ng_i)
        println("  c[$i]:  analytic=$(lpad(round(ag[i],  sigdigits=5), 11))  " *
                "numeric=$(lpad(round(ng_i, sigdigits=5), 11))  " *
                "err=$(round(err, sigdigits=2))")
        flush(stdout)
    end
    println(); flush(stdout)

    # ── GRAPE optimization (L-BFGS) ───────────────────────────────────────────
    iter    = Ref(0)
    t_opt   = Ref(time())
    F_cache = Ref(0.0)   # cache F computed inside grad! for loss()
    F_history = Float64[]  # record F at each gradient evaluation

    # Optim interface: separate loss and grad! (compatible with Optim 2.x)
    function grad!(G, x)
        F, grd = compute_fg(x)
        F_cache[] = F
        push!(F_history, F)
        G .= -grd    # L-BFGS minimizes 1-F; gradient of (1-F) = -gradient of F
        iter[] += 1
        elapsed = round(time() - t_opt[], digits=1)
        if iter[] == 1 || iter[] % 5 == 0
            println("  iter $(lpad(iter[], 4)) | " *
                    "F = $(lpad(round(F, digits=6), 10)) | " *
                    "1-F = $(lpad(round(1.0-F, sigdigits=3), 9)) | " *
                    "t = $(elapsed)s")
            flush(stdout)
        end
    end

    loss(_) = 1.0 - F_cache[]   # re-uses F cached by the last grad! call

    Random.seed!(42)
    control0 = 0.5 * randn(nsteps)   # random initial pulse

    # Evaluate gradient once so F_cache is initialized before the first loss call
    g0 = zeros(nsteps)
    grad!(g0, control0)

    t_opt[] = time()
    println("Starting GRAPE optimization")
    println("  Model  : XXX Heisenberg, N=$N, Jx=Jy=Jz=2π, T=$T, nsteps=$nsteps")
    println("  Task   : |↑↓↓···↓↓⟩ → |↓↓···↓↓↑⟩ (spin-flip transfer)")
    println("  Method : L-BFGS (m=30), max 150 iterations")
    println(); flush(stdout)

    result = Optim.optimize(
        loss,
        grad!,
        control0,
        LBFGS(m=30),
        Optim.Options(
            iterations   = 150,
            g_tol        = 1e-6,
            f_reltol     = 1e-10,
            show_trace   = false,
            store_trace  = true,
            callback     = state -> begin
                F_cur = 1.0 - state.f_x
                if F_cur > 0.9999
                    println("  Early stop: F = $(round(F_cur, digits=6)) > 0.9999")
                    flush(stdout)
                    return true
                end
                return false
            end,
        )
    )

    control_opt = Optim.minimizer(result)
    F_opt       = 1.0 - Optim.minimum(result)
    F_check     = fidelity_only(control_opt)

    println()
    println("══════════════════════════════════════════════════════════")
    println("  Total iterations       = $(iter[])")
    println("  F from optimizer       = $(round(F_opt,   digits=6))")
    println("  F verification (fwd)   = $(round(F_check, digits=6))")
    println("  Infidelity 1-F         = $(round(1-F_check, sigdigits=4))")
    println("  Converged              = $(Optim.converged(result))")
    println("  Total time             = $(round(time()-t_opt[], digits=1))s")
    println(); flush(stdout)

    # ── Final state diagnostics ───────────────────────────────────────────────
    psi_opt = let psi = copy(psi0)
        for n in 1:nsteps
            ctrl = exp(-im * dt * 2control_opt[n] * Sz1)
            psi  = apply(vcat(odd_half, even_half, [ctrl], even_half, odd_half),
                         psi; cutoff, maxdim)
            normalize!(psi)
        end
        psi
    end
    sz_prof = real(expect(psi_opt, "Sz"))
    println("  Sz profile at t=T  : ", round.(sz_prof, digits=3))
    println("  Max bond dim       : ", maxlinkdim(psi_opt))
    println()
    println("  Control pulse stats:")
    println("    max |c|  = $(round(maximum(abs, control_opt), digits=3))")
    println("    rms  c   = $(round(sqrt(mean(control_opt.^2)), digits=3))")

    # ── Plotting ──────────────────────────────────────────────────────────────
    println("\nGenerating plots..."); flush(stdout)
    t_grid = (1:nsteps) .* dt

    # Figure 1: Optimal pulse vs time
    p1 = plot(t_grid, control_opt;
        xlabel="Time", ylabel="Control amplitude c(t)",
        title="Optimal Control Pulse  (N=$N, F=$(round(F_check, digits=6)))",
        legend=false, lw=1.5, color=:steelblue)
    savefig(p1, "pulse.png")
    println("  Saved pulse.png")

    # Figure 2: Infidelity during optimal forward evolution vs time (log scale)
    println("  Computing infidelity vs time..."); flush(stdout)
    infid_t = infidelity_vs_time(control_opt)
    p2 = plot(t_grid, infid_t;
        xlabel="Time", ylabel="1 - F(t)",
        title="Infidelity vs Time  (optimal pulse)",
        legend=false, lw=1.5, color=:crimson,
        yscale=:log10, yminorgrid=true)
    savefig(p2, "infidelity_time.png")
    println("  Saved infidelity_time.png")

    # Figure 3: 1-F vs gradient evaluations (log scale)
    p3 = plot(1:length(F_history), 1.0 .- F_history;
        xlabel="Gradient evaluations", ylabel="1 - F",
        title="GRAPE Convergence  (N=$N)",
        legend=false, lw=1.5, color=:darkorange,
        yscale=:log10, yminorgrid=true)
    savefig(p3, "convergence.png")
    println("  Saved convergence.png")

    (control=control_opt, fidelity=F_check, result=result)
end
