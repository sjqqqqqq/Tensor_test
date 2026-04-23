using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, Printf
include("MPS_TDVP_GRAPE.jl")

# ─ Test 1: per-channel MPO linearity ────────────────────────────────────────
println("── Test 1: MPO linearity ──")
Random.seed!(1)
s = siteinds("SoftBoson", 4)
N = 1
psi0_1, psi_target_1 = default_states(s, N; cutoff=1e-10, maxdim=64)
H_nt   = build_per_channel_mpos(s)
H_list = H_list_from(H_nt)

# Non-product test state: a short TDVP evolution
ctrl_warm = [0.1, 0.2, -0.1, 0.05, -0.2, 0.3, 1.0, 1.0, 1.0]
H_warm    = build_H_step(ctrl_warm, s)
psi_test  = tdvp_step(H_warm, 0.05, psi0_1; cutoff=1e-10, maxdim=64)
println("  psi_test maxlinkdim = ", maxlinkdim(psi_test))

ctrl = randn(9)
lhs  = inner(psi_test', build_H_step(ctrl, s), psi_test)
rhs  = sum(ctrl[k] * inner(psi_test', H_list[k], psi_test) for k in 1:9)
err  = abs(lhs - rhs)
@printf("  |lhs - rhs| = %.3e   lhs=%.6f  rhs=%.6f\n", err, real(lhs), real(rhs))
@assert err < 1e-10 "MPO linearity broken"
println("  ✓ passed\n")

# ─ Test 2: finite-difference gradient check at N=1 ──────────────────────────
println("── Test 2: finite-difference gradient check (N=1) ──")
Random.seed!(42)
nsteps = 100                     # production value
T      = 2π
dt     = T / nsteps
ctrls0 = 0.3 .* randn(nsteps, 9)
ctrls0[:, 7] .+= 1.0; ctrls0[:, 8] .+= 1.0; ctrls0[:, 9] .+= 1.0

cutoff = 1e-10; maxdim = 64

fidelity_only(c) = abs2(inner(psi_target_1,
                              forward_tdvp(c, psi0_1, s; cutoff, maxdim, dt)[end]))

function compute_fg(c; variant::Symbol=:symmetric)
    psis = forward_tdvp(c, psi0_1, s; cutoff, maxdim, dt)
    ov   = inner(psi_target_1, psis[end])
    F    = abs2(ov)
    chi  = copy(psi_target_1)
    grad = zeros(Float64, nsteps, 9)
    for n in nsteps:-1:1
        H_n = build_H_step(@view(c[n, :]), s)
        if variant === :left
            for k in 1:9
                amp = inner(chi', H_list[k], psis[n+1])
                grad[n, k] = 2 * real(conj(ov) * (-im * dt) * amp)
            end
            chi = tdvp_step(H_n, -dt, chi; cutoff, maxdim)
        elseif variant === :right
            chi = tdvp_step(H_n, -dt, chi; cutoff, maxdim)
            for k in 1:9
                amp = inner(chi', H_list[k], psis[n])
                grad[n, k] = 2 * real(conj(ov) * (-im * dt) * amp)
            end
        elseif variant === :symmetric
            # left = ⟨χ̃_n | H_k | ψ_n⟩ (before pullback, psis[n+1])
            lefts = ComplexF64[inner(chi', H_list[k], psis[n+1]) for k in 1:9]
            chi = tdvp_step(H_n, -dt, chi; cutoff, maxdim)
            # right = ⟨χ̃_{n-1} | H_k | ψ_{n-1}⟩ (after pullback, psis[n])
            for k in 1:9
                right = inner(chi', H_list[k], psis[n])
                grad[n, k] = real(conj(ov) * (-im * dt) * (lefts[k] + right))
            end
        else
            error("unknown variant $variant")
        end
    end
    return F, grad
end

for variant in (:left, :right, :symmetric)
    println("\n  variant = :$variant")
    F0, g_an = compute_fg(ctrls0; variant=variant)
    @printf("    baseline F = %.6f\n", F0)
    let h = 1e-5, maxerr = 0.0
        for (n, k) in [(10,3), (50,7), (80,8)]
            cp = copy(ctrls0); cp[n,k] += h
            cm = copy(ctrls0); cm[n,k] -= h
            g_fd = (fidelity_only(cp) - fidelity_only(cm)) / (2h)
            e = abs(g_an[n,k] - g_fd)
            maxerr = max(maxerr, e)
            @printf("    (n=%d,k=%d): analytic=% .6e   fd=% .6e   |Δ|=%.2e\n",
                    n, k, g_an[n,k], g_fd, e)
        end
        @printf("    max |Δ| = %.2e\n", maxerr)
    end
end
