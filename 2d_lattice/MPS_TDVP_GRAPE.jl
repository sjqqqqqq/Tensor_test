using LinearAlgebra, Random, Statistics
using Optim
using JLD2
using Printf

include("soft_boson.jl")

# ─────────────────────────────────────────────────────────────────────────────
# MPO-TDVP GRAPE for 2D soft-core lattice, N a-b pairs
#
# Forward pass : full-MPO TDVP (one MPS stored per step).
# Backward pass: adjoint TDVP (same MPO, negated time) on the costate χ.
# Gradient     : grad[n,k] = 2 Re[ conj(ov) · (-i dt) · ⟨χ_n | H_k | ψ_{n-1}⟩ ]
#                with H_k the fixed per-channel MPO; single inner-with-MPO call.
#
# Replaces the 8×apply + 6×ess per step in MPS_GRAPE.jl with
# 9 × inner(prime, MPO, MPS) + 1 TDVP step per step.
# ─────────────────────────────────────────────────────────────────────────────

# Column order of the 9-control tensor, used everywhere in this file.
const CTRL_KEYS = (:Va1, :Va2, :Va3, :Vb1, :Vb2, :Vb3, :U, :Ja, :Jb)

# Data dir is resolved relative to this script so cwd doesn't matter.
const DATA_DIR = joinpath(@__DIR__, "data")

# ── Per-channel MPOs, built once ─────────────────────────────────────────────
# Generators match exact_GRAPE.jl and MPS_GRAPE.jl:
#   V_{a,k} : Nup@(k+1) − Nup@1       for k ∈ 1,2,3
#   V_{b,k} : Ndn@(k+1) − Ndn@1       for k ∈ 1,2,3
#   U       : Σⱼ Nupdn@j
#   J_a     : Σ ring bonds (Cdagup_j Cup_k + Cup_j Cdagup_k)
#   J_b     : same with dn
function build_per_channel_mpos(s)
    L = length(s)

    # Diagonal potentials
    function os_Vk(name::String, k::Int)
        os = OpSum()
        os += +1.0, name, k+1
        os += -1.0, name, 1
        return os
    end
    os_Va1, os_Va2, os_Va3 = os_Vk("Nup", 1), os_Vk("Nup", 2), os_Vk("Nup", 3)
    os_Vb1, os_Vb2, os_Vb3 = os_Vk("Ndn", 1), os_Vk("Ndn", 2), os_Vk("Ndn", 3)

    os_U = OpSum()
    for j in 1:L; os_U += 1.0, "Nupdn", j; end

    os_Ja = OpSum()
    for (j, k) in ALL_BONDS
        os_Ja += 1.0, "Cdagup", j, "Cup",    k
        os_Ja += 1.0, "Cup",    j, "Cdagup", k
    end
    os_Jb = OpSum()
    for (j, k) in ALL_BONDS
        os_Jb += 1.0, "Cdagdn", j, "Cdn",    k
        os_Jb += 1.0, "Cdn",    j, "Cdagdn", k
    end

    return (Va1 = MPO(os_Va1, s), Va2 = MPO(os_Va2, s), Va3 = MPO(os_Va3, s),
            Vb1 = MPO(os_Vb1, s), Vb2 = MPO(os_Vb2, s), Vb3 = MPO(os_Vb3, s),
            U   = MPO(os_U, s),   Ja  = MPO(os_Ja, s),  Jb  = MPO(os_Jb, s))
end

H_list_from(H_nt) = MPO[H_nt.Va1, H_nt.Va2, H_nt.Va3,
                       H_nt.Vb1, H_nt.Vb2, H_nt.Vb3,
                       H_nt.U,   H_nt.Ja,  H_nt.Jb]

# ── Per-step summed MPO via baked OpSum (cheapest/cleanest on 4 sites) ──────
function build_H_step(ctrl_row::AbstractVector, s; L::Int=length(s))
    Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb = ctrl_row
    os = OpSum()

    # V_a contributions (Nup@(k+1) − Nup@1)
    os += +Va1, "Nup", 2; os += -Va1, "Nup", 1
    os += +Va2, "Nup", 3; os += -Va2, "Nup", 1
    os += +Va3, "Nup", 4; os += -Va3, "Nup", 1
    # V_b contributions
    os += +Vb1, "Ndn", 2; os += -Vb1, "Ndn", 1
    os += +Vb2, "Ndn", 3; os += -Vb2, "Ndn", 1
    os += +Vb3, "Ndn", 4; os += -Vb3, "Ndn", 1
    # On-site interaction
    for j in 1:L; os += U, "Nupdn", j; end
    # Hopping
    for (j, k) in ALL_BONDS
        os += Ja, "Cdagup", j, "Cup",    k; os += Ja, "Cup",    j, "Cdagup", k
        os += Jb, "Cdagdn", j, "Cdn",    k; os += Jb, "Cdn",    j, "Cdagdn", k
    end
    return MPO(os, s)
end

# ── Single TDVP step: U(dt, H) ψ = exp(-i dt H) ψ  ──────────────────────────
# Krylov kwargs (krylovdim, tol) are forwarded to KrylovKit.exponentiate.
# Defaults: krylovdim=30, tol=1e-12 — both wasteful for a 4-site system.
# For ||H||·dt ≈ 0.5, krylovdim=8 converges to tol≈1e-10 in practice.
const TDVP_UPDATER_KWARGS = (; krylovdim = 10, tol = 1e-10)

function tdvp_step(H::MPO, dt::Real, psi::MPS; cutoff=1e-10, maxdim=64,
                   updater_kwargs = TDVP_UPDATER_KWARGS)
    psi_new = tdvp(H, -im*dt, psi;
                   nsweeps=1, cutoff=cutoff, maxdim=maxdim, outputlevel=0,
                   updater_kwargs)
    normalize!(psi_new)
    return psi_new
end

function forward_tdvp(ctrls::AbstractMatrix, psi0::MPS, s; cutoff=1e-10, maxdim=64, dt::Real)
    nsteps = size(ctrls, 1)
    psis = Vector{MPS}(undef, nsteps+1)
    psis[1] = copy(psi0)
    for n in 1:nsteps
        H_n = build_H_step(@view(ctrls[n, :]), s)
        psis[n+1] = tdvp_step(H_n, dt, psis[n]; cutoff, maxdim)
    end
    return psis
end

# ============================================================================
# GRAPE driver
# ============================================================================
"""
Run MPO-TDVP GRAPE for N a-b pairs on the 4-site ring.

Pulses are saved to data/GRAPE_2d_MPS_TDVP_\$(N).jld2 in the same key schema
as the Trotter MPS_GRAPE.jl output.
"""
function run_grape_tdvp(N::Int;
                        T::Float64   = 2π,
                        nsteps::Int  = 100,
                        cutoff       = 1e-10,
                        maxdim       = 64,
                        max_iter     = 100,
                        seed::Int    = 42,
                        save::Bool   = true,
                        verbose::Bool = true,
                        rescale_J::Bool = true)

    dt = T / nsteps
    verbose && println("="^70)
    verbose && println("MPO-TDVP GRAPE — N = $N pair(s),  T=$(round(T,digits=4)),  " *
                       "nsteps=$nsteps,  dt=$(round(dt,digits=5))")
    verbose && println("  NMAX=$(NMAX)  (local dim = $((NMAX+1)^2))")
    verbose && println("="^70)
    if N > NMAX
        @warn "N=$N exceeds NMAX=$NMAX — initial state |a^N@0⟩ would overflow. Set SOFTBOSON_NMAX=$N (or larger) before loading soft_boson.jl."
    end

    # ── Sites, states, per-channel MPOs ─────────────────────────────────────
    s = siteinds("SoftBoson", 4)
    psi0, psi_target = default_states(s, N; cutoff, maxdim)

    H_nt   = build_per_channel_mpos(s)
    H_list = H_list_from(H_nt)

    # ── Forward + cost + gradient ───────────────────────────────────────────
    # Symmetric (left+right average) gradient: O(dt²) accurate.
    #   left  = ⟨χ̃_n     | H_k | ψ_n    ⟩   (chi before pullback, psis[n+1])
    #   right = ⟨χ̃_{n-1} | H_k | ψ_{n-1}⟩   (chi after  pullback, psis[n])
    #   ∂F/∂c_k(n) = Re[ conj(ov) · (-i dt) · (left + right) ]
    # (The factor 2 and the 1/2 symmetrization cancel.)
    function compute_fg_tdvp(ctrls::AbstractMatrix)
        psis = forward_tdvp(ctrls, psi0, s; cutoff, maxdim, dt)
        ov   = inner(psi_target, psis[end])
        F    = abs2(ov)

        chi  = copy(psi_target)
        grad = zeros(Float64, nsteps, 9)

        for n in nsteps:-1:1
            H_n   = build_H_step(@view(ctrls[n, :]), s)
            lefts = ComplexF64[inner(chi', H_list[k], psis[n+1]) for k in 1:9]
            chi   = tdvp_step(H_n, -dt, chi; cutoff, maxdim)
            for k in 1:9
                right       = inner(chi', H_list[k], psis[n])
                grad[n, k]  = real(conj(ov) * (-im * dt) * (lefts[k] + right))
            end
        end
        return F, grad
    end

    fidelity_only(ctrls) = abs2(inner(psi_target, forward_tdvp(ctrls, psi0, s;
                                                               cutoff, maxdim, dt)[end]))

    # ── L-BFGS scaffolding (matches MPS_GRAPE.jl) ───────────────────────────
    iter      = Ref(0)
    t_opt     = Ref(time())
    F_cache   = Ref(0.0)
    F_history = Float64[]

    function grad!(G, x)
        c = reshape(x, nsteps, 9)
        F, grd = compute_fg_tdvp(c)
        F_cache[] = F; push!(F_history, F)
        G .= -vec(grd)
        iter[] += 1
        if verbose && (iter[] == 1 || iter[] % 5 == 0)
            elapsed = round(time() - t_opt[], digits=1)
            println("  iter $(lpad(iter[],4)) | F = $(lpad(round(F,digits=6),10)) " *
                    "| 1-F = $(lpad(round(1-F,sigdigits=3),9)) | t = $(elapsed)s")
            flush(stdout)
        end
    end
    loss(_) = 1.0 - F_cache[]

    # ── Initial controls (warm-start if available) ──────────────────────────
    Random.seed!(seed)
    c0 = load_warm_start(N, nsteps; rescale_J=rescale_J)
    if c0 === nothing
        c0 = 0.3 .* randn(nsteps, 9)
        c0[:, 7] .+= 1.0; c0[:, 8] .+= 1.0; c0[:, 9] .+= 1.0
        verbose && println("  init: random seed=$seed")
    else
        verbose && println("  init: warm-start from GRAPE_2d_MPS_TDVP_$(N-1).jld2" *
                           (rescale_J ? "  (Ja,Jb scaled by √($N/$(N-1)))" : ""))
    end

    g0 = zeros(nsteps*9); grad!(g0, vec(c0))
    t_opt[] = time()
    verbose && println("Starting L-BFGS (m=30, max_iter=$max_iter)..."); flush(stdout)

    result = Optim.optimize(
        loss, grad!, vec(c0),
        LBFGS(m=30),
        Optim.Options(
            iterations  = max_iter,
            g_tol       = 1e-6,
            f_reltol    = 1e-10,
            show_trace  = false,
            store_trace = true,
            callback    = state -> begin
                F_cur = 1.0 - state.f_x
                if F_cur > 0.99
                    verbose && println("  Early stop: F = $(round(F_cur,digits=6)) > 0.99")
                    return true
                end
                return false
            end,
        )
    )

    c_opt   = reshape(Optim.minimizer(result), nsteps, 9)
    F_opt   = 1.0 - Optim.minimum(result)
    F_check = fidelity_only(c_opt)

    if verbose
        println("═"^70)
        println("  N=$N | grad evals=$(iter[]) | F(optim)=$(round(F_opt,digits=6)) " *
                "| F(forward)=$(round(F_check,digits=6))")
        println("  1-F = $(round(1-F_check,sigdigits=4)) | converged=$(Optim.converged(result)) " *
                "| t=$(round(time()-t_opt[],digits=1))s")
    end

    if save
        pad(v) = vcat(v, zero(eltype(v)))
        mkpath(DATA_DIR)
        output_file = joinpath(DATA_DIR, "GRAPE_2d_MPS_TDVP_$(N).jld2")
        jldsave(output_file;
            Npair = N, n = nsteps+1, T = T, dt = dt,
            Va1 = pad(c_opt[:,1]), Va2 = pad(c_opt[:,2]), Va3 = pad(c_opt[:,3]),
            Vb1 = pad(c_opt[:,4]), Vb2 = pad(c_opt[:,5]), Vb3 = pad(c_opt[:,6]),
            U   = pad(c_opt[:,7]), Ja  = pad(c_opt[:,8]), Jb  = pad(c_opt[:,9]),
            fidelity = F_check)
        verbose && println("  Saved $output_file")
    end

    return (controls=c_opt, fidelity=F_check, F_history=F_history,
            result=result, compute_fg_tdvp=compute_fg_tdvp,
            fidelity_only=fidelity_only, psi0=psi0, psi_target=psi_target,
            H_list=H_list, s=s, dt=dt)
end

# ── Warm-start loader ────────────────────────────────────────────────────────
"""
Return a `nsteps × 9` control matrix warm-started from the N-1 TDVP pulse file,
or `nothing` if unavailable. If `rescale_J`, Ja and Jb columns are multiplied
by √(N/(N-1)) to track the √n boson matrix element.
"""
function load_warm_start(N::Int, nsteps::Int; rescale_J::Bool=true)
    N ≤ 1 && return nothing
    # Prefer the matching TDVP pulse; fall back to the Trotter baseline file so
    # the very first TDVP run at each N doesn't need a full TDVP predecessor.
    candidates = [joinpath(DATA_DIR, "GRAPE_2d_MPS_TDVP_$(N-1).jld2"),
                  joinpath(DATA_DIR, "GRAPE_2d_MPS_$(N-1).jld2")]
    f_idx = findfirst(isfile, candidates)
    f_idx === nothing && return nothing
    f = candidates[f_idx]
    println("  warm-start source: $f")
    d = load(f)
    n_prev_saved = Int(d["n"])          # n = nsteps_prev + 1 (pad convention)
    nsteps_prev  = n_prev_saved - 1

    # Stack previous controls, strip the 1-row pad
    keys_order = (:Va1,:Va2,:Va3,:Vb1,:Vb2,:Vb3,:U,:Ja,:Jb)
    prev = hcat([d[String(k)][1:nsteps_prev] for k in keys_order]...)   # nsteps_prev × 9

    # Linear resample onto current nsteps (identity when they match)
    c0 = prev
    if nsteps_prev != nsteps
        t_prev = range(0, 1, length=nsteps_prev)
        t_new  = range(0, 1, length=nsteps)
        c0 = zeros(nsteps, 9)
        for k in 1:9
            @inbounds for (i, tn) in enumerate(t_new)
                # piecewise-linear interpolation
                j = searchsortedlast(t_prev, tn)
                if j ≥ nsteps_prev; c0[i, k] = prev[nsteps_prev, k]
                elseif j < 1;       c0[i, k] = prev[1, k]
                else
                    α = (tn - t_prev[j]) / (t_prev[j+1] - t_prev[j])
                    c0[i, k] = (1-α)*prev[j, k] + α*prev[j+1, k]
                end
            end
        end
    end

    if rescale_J
        # Bosonic hopping matrix element between |n⟩ and |n±1⟩ is √max(n, n+1),
        # so the Rabi angle per step scales as √N·J·dt. To keep the effective
        # rotation ≈ fixed across N, DECREASE J by √((N-1)/N).
        scale = sqrt((N-1) / N)
        c0[:, 8] .*= scale   # Ja
        c0[:, 9] .*= scale   # Jb
    end
    return c0
end

# ============================================================================
# Driver block
# ============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    results = Dict{Int,Any}()
    for N in 1:3
        results[N] = run_grape_tdvp(N)
    end
    println()
    println("─── Summary ────────────────────────────────────────────")
    for N in 1:3
        @printf("  N=%d : F = %.6f\n", N, results[N].fidelity)
    end
end
