using ITensors, ITensorMPS
using LinearAlgebra, Random, Statistics
using Optim
using JLD2

# ── Multi-restart GRAPE for the spin-1-first-site chain ──────────────────────
# Runs several independent random initial guesses in parallel (one per Julia
# thread) to escape the F≈0.966 local-minimum basin found by single starts.
#
# Launch with multiple threads, e.g.:
#   julia -t 12 --project=<proj> 1d_spin_GRAPE_spin1_multistart.jl
#
# Each restart is single-threaded internally (BLAS pinned to 1) so that N_seeds
# restarts run truly concurrently without CPU oversubscription.  Index IDs in
# ITensors use a task-local RNG, and every restart operates on its own copied
# MPS objects (the shared gates/states are read-only), so the threaded loop is
# safe.

BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

let
    # ── System parameters ────────────────────────────────────────────────────
    N      = 11
    Jx     = 2π; Jy = 2π; Jz = 2π
    T      = 15.0
    nsteps = 1500
    dt     = T / nsteps
    cutoff = 1e-10
    maxdim = 100

    iters_per_restart = 200
    seeds = [2024, 1000, 7, 256, 99, 123]

    @assert Jx ≈ Jy "conserve_qns requires Jx = Jy (XXX or XXZ)"

    # ── Site indices, states (shared, read-only) ─────────────────────────────
    s          = siteinds(n -> n == 1 ? "S=1" : "S=1/2", N; conserve_qns=true)
    psi0       = MPS(s, vcat(["Up"], fill("Dn", N - 1)))            # |m=+1,↓···↓⟩
    psi_target = MPS(s, vcat(["Z0"], fill("Dn", N - 2), ["Up"]))   # |m=0,↓···↓↑⟩

    # ── Drift half-step gates (built once, shared read-only) ─────────────────
    function heisenberg_bond(k)
        (Jx/2) * op("S+", s[k]) * op("S-", s[k+1]) +
        (Jx/2) * op("S-", s[k]) * op("S+", s[k+1]) +
        Jz     * op("Sz", s[k]) * op("Sz", s[k+1])
    end
    odd_half      = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
    even_half     = [exp(-im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    adj_odd_half  = [exp(+im * dt/2 * heisenberg_bond(k)) for k in 1:2:N-1]
    adj_even_half = [exp(+im * dt/2 * heisenberg_bond(k)) for k in 2:2:N-1]
    Sz1 = op("Sz", s[1])

    # ── Pure functions of `control` (only read the shared gates/states) ──────
    function forward(control)
        psi = copy(psi0)
        post_ctrl = Vector{MPS}(undef, nsteps)
        for n in 1:nsteps
            psi = apply(vcat(odd_half, even_half), psi; cutoff, maxdim); normalize!(psi)
            psi = apply([exp(-im * dt * control[n] * Sz1)], psi; cutoff, maxdim); normalize!(psi)
            post_ctrl[n] = copy(psi)
            psi = apply(vcat(even_half, odd_half), psi; cutoff, maxdim); normalize!(psi)
        end
        return psi, post_ctrl
    end

    function fidelity_only(control)
        psi = copy(psi0)
        for n in 1:nsteps
            ctrl = exp(-im * dt * control[n] * Sz1)
            psi  = apply(vcat(odd_half, even_half, [ctrl], even_half, odd_half), psi; cutoff, maxdim)
            normalize!(psi)
        end
        return abs2(inner(psi_target, psi))
    end

    function compute_fg(control)
        psi_final, post_ctrl = forward(control)
        overlap = inner(psi_target, psi_final)
        F       = abs2(overlap)
        chi  = copy(psi_target)
        grad = zeros(Float64, nsteps)
        for n in nsteps:-1:1
            c_n = control[n]
            chi = apply(vcat(adj_odd_half, adj_even_half), chi; cutoff, maxdim); normalize!(chi)
            psi_tmp    = copy(post_ctrl[n])
            psi_tmp[1] = noprime(Sz1 * psi_tmp[1])
            grad[n]    = 2real(conj(overlap) * (-im * dt) * inner(chi, psi_tmp))
            chi = apply([exp(+im * dt * c_n * Sz1)], chi; cutoff, maxdim); normalize!(chi)
            chi = apply(vcat(adj_even_half, adj_odd_half), chi; cutoff, maxdim); normalize!(chi)
        end
        return F, grad
    end

    # ── One full GRAPE optimization from a given seed (thread-local state) ───
    print_lock = ReentrantLock()
    function run_one(seed)
        control0 = 0.5 * randn(MersenneTwister(seed), nsteps)
        F_cache   = Ref(0.0)
        local_it  = Ref(0)
        function grad!(G, x)
            F, grd = compute_fg(x)
            F_cache[] = F
            G .= -grd
            local_it[] += 1
            if local_it[] == 1 || local_it[] % 25 == 0
                lock(print_lock) do
                    println("  [seed $(lpad(seed,5))] eval $(lpad(local_it[],4)) | F = $(round(F, digits=6))")
                    flush(stdout)
                end
            end
        end
        loss(_) = 1.0 - F_cache[]
        g0 = zeros(nsteps); grad!(g0, control0)   # init F_cache
        result = Optim.optimize(loss, grad!, control0, LBFGS(m=30),
            Optim.Options(iterations=iters_per_restart, g_tol=1e-6, f_reltol=1e-10,
                callback = st -> (1.0 - st.f_x) > 0.9999))
        c_opt = Optim.minimizer(result)
        return c_opt, fidelity_only(c_opt), Optim.converged(result), local_it[]
    end

    # ── JIT warmup (single-threaded; compiles before the parallel loop) ──────
    println("Threads available: $(Threads.nthreads()) | restarts: $(length(seeds)) | iters/restart: $iters_per_restart")
    println("JIT warmup..."); flush(stdout)
    compute_fg(zeros(nsteps)); fidelity_only(zeros(nsteps))
    println("Warmup done. Launching parallel restarts...\n"); flush(stdout)

    # ── Parallel multi-restart ────────────────────────────────────────────────
    controls = Vector{Vector{Float64}}(undef, length(seeds))
    Fs       = Vector{Float64}(undef, length(seeds))
    convs    = Vector{Bool}(undef, length(seeds))
    nevals   = Vector{Int}(undef, length(seeds))
    t0 = time()
    Threads.@threads for i in eachindex(seeds)
        c, F, cv, ne = run_one(seeds[i])
        controls[i] = c; Fs[i] = F; convs[i] = cv; nevals[i] = ne
        lock(print_lock) do
            println(">>> seed $(lpad(seeds[i],5)) DONE | F = $(round(F, digits=6)) | converged=$cv | evals=$ne")
            flush(stdout)
        end
    end
    wall = time() - t0

    # ── Summary ─────────────────────────────────────────────────────────────
    order = sortperm(Fs; rev=true)
    println("\n══════════════════════ multi-restart summary ══════════════════════")
    println("  wall time = $(round(wall, digits=1))s  ($(length(seeds)) restarts in parallel)")
    println("  rank  seed     F          1-F        converged  evals")
    for (r, i) in enumerate(order)
        println("  $(lpad(r,3))  $(lpad(seeds[i],5))  $(lpad(round(Fs[i],digits=6),9))  " *
                "$(lpad(round(1-Fs[i],sigdigits=3),9))  $(lpad(string(convs[i]),9))  $(nevals[i])")
    end

    best   = order[1]
    c_best = controls[best]; F_best = Fs[best]
    println("\n  BEST: seed $(seeds[best]), F = $(round(F_best, digits=6))")

    # Final-state diagnostics for the best pulse
    psi_b = let psi = copy(psi0)
        for n in 1:nsteps
            psi = apply(vcat(odd_half, even_half, [exp(-im*dt*c_best[n]*Sz1)], even_half, odd_half),
                        psi; cutoff, maxdim); normalize!(psi)
        end
        psi
    end
    println("  Sz profile at t=T : ", round.(real(expect(psi_b, "Sz")), digits=3))
    println("  pulse: max|c| = $(round(maximum(abs,c_best),digits=3)), rms = $(round(sqrt(mean(c_best.^2)),digits=3))")

    # ── Persist all restarts; update main cache only if best beats it ────────
    ms_file = joinpath(@__DIR__, "1d_spin_spin1_multistart.jld2")
    jldsave(ms_file; seeds, controls, Fs, convs, N, T, nsteps)
    println("\n  Saved all restarts → $ms_file")

    cache_file = joinpath(@__DIR__, "1d_spin_spin1.jld2")
    prev_F = isfile(cache_file) ? load(cache_file, "F_check") : -Inf
    if F_best > prev_F
        jldsave(cache_file; control_opt=c_best, F_opt=F_best, F_check=F_best, N, T, nsteps)
        println("  New best beats cached F=$(round(prev_F,digits=6)) → updated $cache_file")
    else
        println("  Cached F=$(round(prev_F,digits=6)) still best; cache unchanged.")
    end

    (seeds=seeds, Fs=Fs, controls=controls)
end
