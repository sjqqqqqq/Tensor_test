# 2d_lattice_GRAPE.jl
# GRAPE optimal control for 2D two-particle lattice using Trotter propagation
# State transfer: |0,1⟩ → (|0,1⟩ + |2,3⟩)/√2
#
# System: 2 distinguishable particles (a, b) each hopping on a 4-site ring.
# Hilbert space: 4 ⊗ 4 = 16 dimensions, basis |s,t⟩ = |site_a, site_b⟩.
#
# Full Hamiltonian:
#   H(t) = Va1(t)·H_Va1 + Va2(t)·H_Va2 + Va3(t)·H_Va3
#         + Vb1(t)·H_Vb1 + Vb2(t)·H_Vb2 + Vb3(t)·H_Vb3
#         + U(t)·H_U + Ja(t)·H_Ja + Jb(t)·H_Jb
#
# Trotter split:
#   H₁ = Va·terms + Vb·terms + U·H_U   (diagonal in |s,t⟩ basis)
#   H_Ja = kron(Γ, I₄),  H_Jb = kron(I₄, Γ)
#
#   Since [H_Ja, H_Jb] = 0 exactly, exp(-i(Ja·H_Ja + Jb·H_Jb)dt)
#   = exp(-iJa·H_Ja·dt) · exp(-iJb·H_Jb·dt)  (no Trotter error for this split).
#
#   2nd-order scheme:
#   exp(-iHdt) ≈ exp(-iH₁dt/2) · exp(-iJa·H_Ja·dt) · exp(-iJb·H_Jb·dt) · exp(-iH₁dt/2)

using LinearAlgebra
using Printf
using Random
using Optim

# ============================================================================
# System definition
# ============================================================================

const Gamma4 = Float64[0 1 1 0;
                       1 0 0 1;
                       1 0 0 1;
                       0 1 1 0]

const I4 = Matrix{ComplexF64}(I, 4, 4)

function projector4(s)
    e = zeros(ComplexF64, 4)
    e[s + 1] = 1.0
    return e * e'
end

# Particle projectors (diagonal in the |s,t⟩ basis)
const _Pa = [kron(projector4(s), I4) for s in 0:3]
const _Pb = [kron(I4, projector4(t)) for t in 0:3]

# Relative-potential control matrices (all diagonal)
const H_Va1_mat = _Pa[2] - _Pa[1]   # particle a: site 1 relative to site 0
const H_Va2_mat = _Pa[3] - _Pa[1]   # particle a: site 2 relative to site 0
const H_Va3_mat = _Pa[4] - _Pa[1]   # particle a: site 3 relative to site 0
const H_Vb1_mat = _Pb[2] - _Pb[1]
const H_Vb2_mat = _Pb[3] - _Pb[1]
const H_Vb3_mat = _Pb[4] - _Pb[1]

# Interaction (diagonal): same-site occupation
const H_U_mat = Matrix{ComplexF64}(sum(kron(projector4(s), projector4(s)) for s in 0:3))

# Hopping matrices (non-diagonal)
const H_Ja_mat = Matrix{ComplexF64}(kron(Gamma4, I4))
const H_Jb_mat = Matrix{ComplexF64}(kron(I4, Gamma4))

# Precomputed diagonal vectors (all real since matrices are real diagonal)
const dVa1 = real(diag(H_Va1_mat))
const dVa2 = real(diag(H_Va2_mat))
const dVa3 = real(diag(H_Va3_mat))
const dVb1 = real(diag(H_Vb1_mat))
const dVb2 = real(diag(H_Vb2_mat))
const dVb3 = real(diag(H_Vb3_mat))
const dU_d = real(diag(H_U_mat))

# State index for |site_a, site_b⟩  (sites 0-indexed)
idx2d(s, t) = s * 4 + t + 1

# ============================================================================
# Trotter propagation utilities
# ============================================================================

"""Eigendecompose a Hermitian matrix."""
function eigen_decomp(H)
    F = eigen(Hermitian(Matrix(H)))
    return F.values, F.vectors
end

"""
Diagonal vector of exp(-i·H₁·dt) where
H₁ = Va1·H_Va1 + ... + U·H_U  is diagonal.
Controls: (Va1,Va2,Va3,Vb1,Vb2,Vb3,U), time step dt.
"""
function diag_exp(Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt)
    h = Va1 .* dVa1 .+ Va2 .* dVa2 .+ Va3 .* dVa3 .+
        Vb1 .* dVb1 .+ Vb2 .* dVb2 .+ Vb3 .* dVb3 .+
        U   .* dU_d
    return exp.(-im * dt .* h)
end

"""
Single 2nd-order Trotter step:
  ψ → D1 · U_Jb · U_Ja · D1 · ψ
where D1 = exp(-iH₁·dt/2), U_Ja = exp(-iJa·H_Ja·dt), U_Jb = exp(-iJb·H_Jb·dt).
"""
function trotter_step2d(psi, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt,
                        evals_a, evecs_a, evals_b, evecs_b)
    d1 = diag_exp(Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt/2)
    psi = d1 .* psi                                                        # D1 (first half)
    psi = evecs_a * (Diagonal(exp.(-im*Ja*dt.*evals_a)) * (evecs_a'*psi)) # U_Ja
    psi = evecs_b * (Diagonal(exp.(-im*Jb*dt.*evals_b)) * (evecs_b'*psi)) # U_Jb
    psi = d1 .* psi                                                        # D1 (second half)
    return psi
end

"""Forward propagation; ctrls is (num_steps × 9) matrix."""
function trotter_fwd(psi0, ctrls, dt, evals_a, evecs_a, evals_b, evecs_b)
    psi = copy(psi0)
    for n in 1:size(ctrls,1)-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = trotter_step2d(psi,Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb,dt,
                             evals_a,evecs_a,evals_b,evecs_b)
    end
    return psi
end

"""Forward propagation storing all intermediate states (for GRAPE)."""
function trotter_fwd_store(psi0, ctrls, dt, evals_a, evecs_a, evals_b, evecs_b)
    num_steps = size(ctrls,1)
    states = Vector{Vector{ComplexF64}}(undef, num_steps)
    psi = copy(psi0)
    states[1] = copy(psi)
    for n in 1:num_steps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = trotter_step2d(psi,Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb,dt,
                             evals_a,evecs_a,evals_b,evecs_b)
        states[n+1] = copy(psi)
    end
    return states
end

"""
Backward propagation of costate, storing all states.
Adjoint of (D1·U_Jb·U_Ja·D1) is (D1†·U_Ja†·U_Jb†·D1†).
"""
function trotter_bwd_store(chi_T, ctrls, dt, evals_a, evecs_a, evals_b, evecs_b)
    num_steps = size(ctrls,1)
    costates = Vector{Vector{ComplexF64}}(undef, num_steps)
    chi = copy(chi_T)
    costates[num_steps] = copy(chi)
    for n in num_steps-1:-1:1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        d1 = diag_exp(Va1,Va2,Va3,Vb1,Vb2,Vb3,U,dt/2)
        pa = exp.(-im*Ja*dt.*evals_a)
        pb = exp.(-im*Jb*dt.*evals_b)
        chi = conj.(d1) .* chi                                                   # D1†
        chi = evecs_b * (Diagonal(conj.(pb)) * (evecs_b' * chi))                 # U_Jb†
        chi = evecs_a * (Diagonal(conj.(pa)) * (evecs_a' * chi))                 # U_Ja†
        chi = conj.(d1) .* chi                                                   # D1†
        costates[n] = copy(chi)
    end
    return costates
end

# ============================================================================
# GRAPE gradient computation
# ============================================================================

"""
Compute gradients of F = |⟨ψ_target|ψ_T⟩|² w.r.t. all 9 controls.
Returns a (num_steps × 9) matrix.

Control ordering per time step: [Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb]

GRAPE formula:
  ∂F/∂u_n = 2 Re{ overlap* · ⟨χ_{n+1}| ∂U_n/∂u_n |ψ_n⟩ }
where overlap = ⟨ψ_target|ψ_T⟩, χ_{n+1} = backward-propagated costate.

Trotter step: U_n = D1_R · U_Jb · U_Ja · D1_L  (D1_R = D1_L = D1)

Intermediate states for step n:
  ψ1 = D1·ψ_n,  ψ2 = U_Ja·ψ1,  ψ3 = U_Jb·ψ2

Gradient terms:
  Ja:  ∂U_n/∂Ja·ψ_n = D1·U_Jb·(∂U_Ja/∂Ja·ψ1)
  Jb:  ∂U_n/∂Jb·ψ_n = D1·(∂U_Jb/∂Jb·ψ2)
  ck (diagonal): ∂U_n/∂ck·ψ_n = (∂D1/∂ck)·ψ3 + D1·U_Jb·U_Ja·(∂D1/∂ck·ψ_n)
                where ∂D1/∂ck·v = -i(dt/2)·hk .* (d1 .* v)
"""
function compute_grads(psi_states, chi_states, ctrls, dt,
                       evals_a, evecs_a, evals_b, evecs_b, overlap)
    num_steps = size(ctrls,1)
    grads = zeros(num_steps, 9)

    for n in 1:num_steps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        ψn   = psi_states[n]
        χnp1 = chi_states[n+1]

        d1 = diag_exp(Va1,Va2,Va3,Vb1,Vb2,Vb3,U,dt/2)
        pa = exp.(-im*Ja*dt.*evals_a)
        pb = exp.(-im*Jb*dt.*evals_b)

        # Intermediate states
        ψ1 = d1 .* ψn
        ψ2 = evecs_a * (Diagonal(pa) * (evecs_a' * ψ1))
        ψ3 = evecs_b * (Diagonal(pb) * (evecs_b' * ψ2))

        # --- Gradient w.r.t. Ja ---
        # ∂U_Ja/∂Ja · ψ1 = evecs_a · Diag(-i·dt·λ·phase_a) · evecs_a' · ψ1
        dJa_ψ1 = evecs_a * (Diagonal(-im*dt.*evals_a.*pa) * (evecs_a' * ψ1))
        gJa = d1 .* (evecs_b * (Diagonal(pb) * (evecs_b' * dJa_ψ1)))
        grads[n, 8] = 2 * real(conj(overlap) * dot(χnp1, gJa))

        # --- Gradient w.r.t. Jb ---
        dJb_ψ2 = evecs_b * (Diagonal(-im*dt.*evals_b.*pb) * (evecs_b' * ψ2))
        gJb = d1 .* dJb_ψ2
        grads[n, 9] = 2 * real(conj(overlap) * dot(χnp1, gJb))

        # --- Gradients w.r.t. diagonal controls ---
        # ∂D1/∂ck · v = -i(dt/2) · hk .* (d1 .* v)
        diag_vecs = (dVa1, dVa2, dVa3, dVb1, dVb2, dVb3, dU_d)
        for (k, hk) in enumerate(diag_vecs)
            # Right D1: ∂D1/∂ck · ψ3
            right = (-im*dt/2) .* hk .* (d1 .* ψ3)
            # Left D1: D1·U_Jb·U_Ja · (∂D1/∂ck · ψ_n)
            dD1_ψn = (-im*dt/2) .* hk .* (d1 .* ψn)
            left = d1 .* (evecs_b * (Diagonal(pb) * (evecs_b' * (evecs_a * (Diagonal(pa) * (evecs_a' * dD1_ψn))))))
            grads[n, k] = 2 * real(conj(overlap) * dot(χnp1, right .+ left))
        end
    end

    return grads
end

# ============================================================================
# GRAPE optimization (L-BFGS via Optim.jl)
# ============================================================================

"""
Run GRAPE optimization for the 2D lattice state transfer.

Arguments:
  psi0, psi_target : initial and target states (length-16 vectors)
  T                : total evolution time
  num_steps        : number of time steps (including t=0 and t=T)
  ctrls0           : (num_steps × 9) initial controls [Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb]
  max_iter, tol    : L-BFGS stopping criteria

Returns (ctrls_opt, final_fidelity).
"""
function grape_2d(psi0, psi_target, T, num_steps;
                  ctrls0=nothing, max_iter=500, tol=1e-4, verbose=true)

    dt = T / (num_steps - 1)
    evals_a, evecs_a = eigen_decomp(H_Ja_mat)
    evals_b, evecs_b = eigen_decomp(H_Jb_mat)

    ψ0 = Vector{ComplexF64}(psi0)
    ψt = Vector{ComplexF64}(psi_target)

    if isnothing(ctrls0)
        ctrls = zeros(num_steps, 9)
        ctrls[:, 7] .= 1.0   # U
        ctrls[:, 8] .= 1.0   # Ja
        ctrls[:, 9] .= 1.0   # Jb
    else
        ctrls = copy(ctrls0)
    end

    function objective(x)
        c = reshape(x, num_steps, 9)
        ψf = trotter_fwd(ψ0, c, dt, evals_a, evecs_a, evals_b, evecs_b)
        return 1.0 - abs2(dot(ψt, ψf))
    end

    function gradient!(g, x)
        c = reshape(x, num_steps, 9)
        states   = trotter_fwd_store(ψ0, c, dt, evals_a, evecs_a, evals_b, evecs_b)
        overlap  = dot(ψt, states[end])
        costates = trotter_bwd_store(ψt, c, dt, evals_a, evecs_a, evals_b, evecs_b)
        grads    = compute_grads(states, costates, c, dt, evals_a, evecs_a, evals_b, evecs_b, overlap)
        g .= -vec(grads)   # negate: we minimise infidelity
    end

    x0 = vec(ctrls)
    iter_count = Ref(0)

    function cb(state)
        iter_count[] += 1
        if verbose && (iter_count[] == 1 || iter_count[] % 10 == 0)
            @printf("Iter %4d: fidelity = %.8f\n", iter_count[], 1.0 - state.value)
        end
        return false
    end

    result = optimize(
        objective, gradient!, x0,
        LBFGS(m=20),
        Optim.Options(
            iterations  = max_iter,
            g_tol       = tol * 1e-2,
            f_reltol    = tol * 1e-2,
            show_trace  = false,
            callback    = cb
        )
    )

    ctrls_opt      = reshape(Optim.minimizer(result), num_steps, 9)
    final_fidelity = 1.0 - Optim.minimum(result)

    if verbose
        println("\nL-BFGS completed:")
        println("  Iterations: $(Optim.iterations(result))")
        println("  Converged:  $(Optim.converged(result))")
    end

    return ctrls_opt, final_fidelity
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("="^60)
    println("GRAPE (Trotter) — 2D Two-Particle Lattice")
    println("="^60)
    println("4-site lattice × 2 particles → 16-dimensional Hilbert space")
    println("Trotter: exp(-iH₁dt/2)·exp(-iJa·H_Ja·dt)·exp(-iJb·H_Jb·dt)·exp(-iH₁dt/2)")
    println("Note: [H_Ja, H_Jb] = 0, so hopping split is exact.")

    # States
    psi0 = zeros(ComplexF64, 16);  psi0[idx2d(0,1)] = 1.0
    psi_target = zeros(ComplexF64, 16)
    psi_target[idx2d(0,1)] = 1/sqrt(2)
    psi_target[idx2d(2,3)] = 1/sqrt(2)

    println("Initial:  |0,1⟩")
    println("Target:   (|0,1⟩ + |2,3⟩)/√2")

    T = 2π
    num_steps = 201
    dt = T / (num_steps - 1)
    @printf("Time: T = %.4f,  steps = %d,  dt = %.4f\n\n", T, num_steps, dt)

    # Initial guess for controls [Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb]
    Random.seed!(42)
    t_arr = collect(range(0, T, length=num_steps))
    ctrls0 = zeros(num_steps, 9)
    for k in 1:6
        ctrls0[:, k] .= 0.1 * randn(num_steps)             # small random site potentials
    end
    ctrls0[:, 7] .= 1.0 .+ 0.1 * randn(num_steps)          # U
    ctrls0[:, 8] .= 1.0 .+ 0.2*sin.(2π.*t_arr./T) .+ 0.1*randn(num_steps)  # Ja
    ctrls0[:, 9] .= 1.0 .+ 0.2*cos.(2π.*t_arr./T) .+ 0.1*randn(num_steps)  # Jb

    println("Starting GRAPE optimization...")
    println("-"^60)

    ctrls_opt, final_fidelity = grape_2d(
        psi0, psi_target, T, num_steps;
        ctrls0=ctrls0, max_iter=500, tol=1e-4, verbose=true
    )

    println("-"^60)
    @printf("Final fidelity: %.8f\n", final_fidelity)

    # Verify
    evals_a, evecs_a = eigen_decomp(H_Ja_mat)
    evals_b, evecs_b = eigen_decomp(H_Jb_mat)
    ψf = trotter_fwd(Vector{ComplexF64}(psi0), ctrls_opt, dt, evals_a, evecs_a, evals_b, evecs_b)
    println("\nFinal state analysis:")
    @printf("  |⟨ψ_target|ψ_final⟩|² = %.8f\n", abs2(dot(psi_target, ψf)))
    @printf("  |ψ[0,1]|²             = %.6f\n",  abs2(ψf[idx2d(0,1)]))
    @printf("  |ψ[2,3]|²             = %.6f\n",  abs2(ψf[idx2d(2,3)]))

    # Save controls
    ctrl_names = ["Va1","Va2","Va3","Vb1","Vb2","Vb3","U","Ja","Jb"]
    println("\nSaving controls...")
    for (k, name) in enumerate(ctrl_names)
        open("$(name)_2d_opt.txt", "w") do f
            for v in ctrls_opt[:,k]; println(f, v); end
        end
    end
    println("Saved to <name>_2d_opt.txt  ($(length(ctrl_names)) files)")

    return ctrls_opt, final_fidelity
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
