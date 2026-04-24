using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Printf, Random, LinearAlgebra, Statistics

# ─────────────────────────────────────────────────────────────────────────────
# Forward-propagator benchmark for the 4-site soft-core two-species lattice.
# One time-varying pulse (seeded random controls), four propagators, N ∈ 1..4.
#
# Reports per (method, N): median wall time over warm trials, peak MPS bond
# dim, and |F − F_dense| to measure discretization / truncation error.
# ─────────────────────────────────────────────────────────────────────────────

# NMAX must be ≥ max N so |aᴺ@0⟩ fits on a SoftBoson site.
const TARGET_NS = 1:4
ENV["SOFTBOSON_NMAX"] = string(maximum(TARGET_NS))

# Load dense engine first (does not touch SoftBoson), then TDVP engine
# (which includes soft_boson.jl and pulls NMAX from ENV).
include("exact_GRAPE.jl")        # dense: build_system, trotter_step2d, eigen_decomp, default_states(sys)
include("MPS_TDVP_GRAPE.jl")     # MPS: soft_boson.jl + TDVP, default_states(s, N)

# The MPS Trotter step lives in MPS_sim.jl; copy it here to avoid re-including
# soft_boson.jl (would double-define consts).
function mps_trotter_step(s, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, step_dt;
                          cutoff, maxdim)
    d1h = make_onsite_gates(s, Va1,Va2,Va3, Vb1,Vb2,Vb3, U, step_dt/2)
    psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi)
    psi = apply(hop_gates_a(s, Ja, step_dt), psi; cutoff, maxdim); normalize!(psi)
    psi = apply(hop_gates_b(s, Jb, step_dt), psi; cutoff, maxdim); normalize!(psi)
    psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi)
    return psi
end

# ── Reference pulse: identical across methods and N ────────────────────────
function make_pulse(nsteps::Int, T::Float64; seed::Int=42)
    Random.seed!(seed)
    t_arr = collect(range(0, T, length=nsteps))
    c = zeros(nsteps, 9)
    for k in 1:6; c[:,k] .= 0.1 * randn(nsteps); end
    c[:,7] .= 1.0 .+ 0.1 * randn(nsteps)
    c[:,8] .= 1.0 .+ 0.2 * sin.(2π .* t_arr ./ T) .+ 0.1 * randn(nsteps)
    c[:,9] .= 1.0 .+ 0.2 * cos.(2π .* t_arr ./ T) .+ 0.1 * randn(nsteps)
    return c
end

# ── Propagator wrappers ───────────────────────────────────────────────────
# All return (F_final::Float64, peak_bond::Union{Int,Nothing}).

function fwd_dense(ctrls, dt, sys, psi0, psi_target)
    evals_a, evecs_a = eigen_decomp(sys.H_Ja_mat)
    evals_b, evecs_b = eigen_decomp(sys.H_Jb_mat)
    psi = Vector{ComplexF64}(psi0)
    for n in 1:size(ctrls,1)
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = trotter_step2d(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt,
                             evals_a,evecs_a,evals_b,evecs_b)
    end
    return abs2(dot(psi_target, psi)), nothing
end

function fwd_mps_trotter(ctrls, dt, s, psi0, psi_target; cutoff, maxdim)
    psi = copy(psi0)
    peak = maxlinkdim(psi)
    for n in 1:size(ctrls,1)
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = mps_trotter_step(s, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt;
                               cutoff, maxdim)
        peak = max(peak, maxlinkdim(psi))
    end
    return abs2(inner(psi_target, psi)), peak
end

function fwd_mps_tdvp(ctrls, dt, s, psi0, psi_target; cutoff, maxdim,
                      backend::String="exponentiate",
                      updater_kwargs=(; krylovdim=10, tol=1e-10))
    psi = copy(psi0)
    peak = maxlinkdim(psi)
    for n in 1:size(ctrls,1)
        H_n = build_H_step(@view(ctrls[n,:]), s)
        psi = tdvp(H_n, -im*dt, psi;
                   nsweeps=1, cutoff, maxdim, outputlevel=0,
                   updater_backend=backend, updater_kwargs)
        normalize!(psi)
        peak = max(peak, maxlinkdim(psi))
    end
    return abs2(inner(psi_target, psi)), peak
end

# ── Driver ────────────────────────────────────────────────────────────────
const T      = 2π
const NSTEPS = 100
const DT     = T / NSTEPS
const CUTOFF = 1e-10
const MAXDIM = 64
const TRIALS = 3           # warm trials (plus 1 warmup discard)
const PULSE  = make_pulse(NSTEPS, T; seed=42)

function trial_median(f, args...; trials=TRIALS)
    ts = Float64[]
    F_last = 0.0
    bond_last = nothing
    for _ in 1:trials
        t0 = time()
        F, bond = f(args...)
        push!(ts, time() - t0)
        F_last = F; bond_last = bond
    end
    return (F=F_last, t_med=median(ts), t_min=minimum(ts), bond=bond_last)
end

println("━"^74)
println("Forward propagator benchmark — pulse: seeded random, T=$(round(T,digits=3)), nsteps=$NSTEPS, dt=$(round(DT,digits=4))")
println("NMAX=$(NMAX)  (local MPS dim = $((NMAX+1)^2))  maxdim=$MAXDIM  cutoff=$CUTOFF")
println("━"^74)

for N in TARGET_NS
    println("\n── N = $N ────────────────────────────────────────────────────────────────")

    # Systems/states for this N
    sys_dense = build_system(N)
    psi0_d, psi_t_d = default_states(sys_dense)

    s_mps = siteinds("SoftBoson", 4)
    psi0_m, psi_t_m = default_states(s_mps, N; cutoff=CUTOFF, maxdim=MAXDIM)

    # JIT warm-up for each method (discard timing)
    fwd_dense(PULSE, DT, sys_dense, psi0_d, psi_t_d)
    fwd_mps_trotter(PULSE, DT, s_mps, psi0_m, psi_t_m; cutoff=CUTOFF, maxdim=MAXDIM)
    fwd_mps_tdvp(PULSE, DT, s_mps, psi0_m, psi_t_m; cutoff=CUTOFF, maxdim=MAXDIM)
    fwd_mps_tdvp(PULSE, DT, s_mps, psi0_m, psi_t_m; cutoff=CUTOFF, maxdim=MAXDIM,
                 backend="applyexp", updater_kwargs=(; maxiter=10, tol=1e-10))

    # Timed trials
    r_d = trial_median(fwd_dense, PULSE, DT, sys_dense, psi0_d, psi_t_d)
    r_t = trial_median((a,b,c,d,e)->fwd_mps_trotter(a,b,c,d,e; cutoff=CUTOFF, maxdim=MAXDIM),
                       PULSE, DT, s_mps, psi0_m, psi_t_m)
    r_k = trial_median((a,b,c,d,e)->fwd_mps_tdvp(a,b,c,d,e; cutoff=CUTOFF, maxdim=MAXDIM),
                       PULSE, DT, s_mps, psi0_m, psi_t_m)
    r_a = trial_median((a,b,c,d,e)->fwd_mps_tdvp(a,b,c,d,e; cutoff=CUTOFF, maxdim=MAXDIM,
                                                 backend="applyexp",
                                                 updater_kwargs=(; maxiter=10, tol=1e-10)),
                       PULSE, DT, s_mps, psi0_m, psi_t_m)

    F_ref = r_d.F
    function row(label, r)
        err = abs(r.F - F_ref)
        bondstr = isnothing(r.bond) ? "  —" : lpad(r.bond, 3)
        @printf("  %-22s  F=%.6f  |ΔF|=%.2e  bond=%s  t_med=%.3fs  t_min=%.3fs\n",
                label, r.F, err, bondstr, r.t_med, r.t_min)
    end
    row("dense Trotter",       r_d)
    row("MPS Trotter",         r_t)
    row("MPS TDVP-exponentiate", r_k)
    row("MPS TDVP-applyexp",   r_a)
end

println("\n━"^74)
println("Done.")
