# BM_GRAPE_trotter.jl
# GRAPE optimal control for Bose-Hubbard model with Trotter propagation
# Transfer from |N,0,...,0> to (|N,0,...,0> + |0,...,0,N>)/sqrt(2)
#
# Uses second-order Trotterization:
# exp(-iHdt) ≈ exp(-iH₁dt/2) · exp(-iH₂dt) · exp(-iH₁dt/2)
# where H₁ = U·H_int + Δ·H_sz (diagonal), H₂ = J·H_hop

using LinearAlgebra
using Combinatorics
using Printf
using Random
using Optim

include("BM_fuctions.jl")

# ============================================================================
# Trotter propagation utilities
# ============================================================================

"""
Precompute eigendecomposition of H_hop for efficient Trotter steps.
Returns eigenvalues and eigenvectors.
"""
function precompute_hop_eigen(H_hop)
    H_hop_dense = Matrix(H_hop)
    F = eigen(Hermitian(H_hop_dense))
    return F.values, F.vectors
end

"""
Compute exp(-i * J * H_hop * dt) using precomputed eigendecomposition.
"""
function exp_hop(J, dt, hop_evals, hop_evecs)
    # exp(-i*J*H_hop*dt) = V * diag(exp(-i*J*λ*dt)) * V†
    phases = exp.(-im * J * dt .* hop_evals)
    return hop_evecs * Diagonal(phases) * hop_evecs'
end

"""
Compute exp(-i * (U*H_int + Δ*H_sz) * dt) for diagonal Hamiltonians.
H_int and H_sz are Diagonal matrices.
"""
function exp_diagonal(U, Delta, dt, H_int_diag, H_sz_diag)
    # H₁ = U*H_int + Δ*H_sz is diagonal
    phases = exp.(-im * dt .* (U .* H_int_diag .+ Delta .* H_sz_diag))
    return Diagonal(phases)
end

"""
Single second-order Trotter step:
exp(-iHdt) ≈ exp(-iH₁dt/2) · exp(-iH₂dt) · exp(-iH₁dt/2)
where H₁ = U·H_int + Δ·H_sz, H₂ = J·H_hop
"""
function trotter_step(psi, J, U, Delta, dt, H_int_diag, H_sz_diag, hop_evals, hop_evecs)
    # First half-step: exp(-iH₁dt/2)
    D1 = exp_diagonal(U, Delta, dt/2, H_int_diag, H_sz_diag)
    psi = D1 * psi

    # Full step: exp(-iH₂dt)
    U_hop = exp_hop(J, dt, hop_evals, hop_evecs)
    psi = U_hop * psi

    # Second half-step: exp(-iH₁dt/2)
    psi = D1 * psi

    return psi
end

"""
Propagate state from t=0 to t=T using Trotter decomposition.
"""
function trotter_propagate(psi0, J_ctrl, U_ctrl, Delta_ctrl, dt,
                           H_int_diag, H_sz_diag, hop_evals, hop_evecs)
    psi = copy(psi0)
    num_steps = length(J_ctrl)

    for n in 1:num_steps-1
        # Use control values at step n for propagation from t_n to t_{n+1}
        psi = trotter_step(psi, J_ctrl[n], U_ctrl[n], Delta_ctrl[n], dt,
                          H_int_diag, H_sz_diag, hop_evals, hop_evecs)
    end

    return psi
end

"""
Propagate and store all intermediate states (for GRAPE gradient computation).
Returns array of states at each time step.
"""
function trotter_propagate_store(psi0, J_ctrl, U_ctrl, Delta_ctrl, dt,
                                  H_int_diag, H_sz_diag, hop_evals, hop_evecs)
    num_steps = length(J_ctrl)
    d = length(psi0)

    # Store states: psi_states[n] is state at time t_{n-1}
    psi_states = Vector{Vector{ComplexF64}}(undef, num_steps)

    psi = copy(psi0)
    psi_states[1] = copy(psi)

    for n in 1:num_steps-1
        psi = trotter_step(psi, J_ctrl[n], U_ctrl[n], Delta_ctrl[n], dt,
                          H_int_diag, H_sz_diag, hop_evals, hop_evecs)
        psi_states[n+1] = copy(psi)
    end

    return psi_states
end

"""
Backward propagate costate (for GRAPE gradient).
"""
function trotter_backward_propagate_store(chi_T, J_ctrl, U_ctrl, Delta_ctrl, dt,
                                          H_int_diag, H_sz_diag, hop_evals, hop_evecs)
    num_steps = length(J_ctrl)

    # Store costates: chi_states[n] is costate at time t_{n-1}
    chi_states = Vector{Vector{ComplexF64}}(undef, num_steps)

    chi = copy(chi_T)
    chi_states[num_steps] = copy(chi)

    # Backward propagation uses adjoint of forward propagator
    for n in num_steps-1:-1:1
        # Adjoint of Trotter step (reverse order, conjugate)
        # Forward: D1 * U_hop * D1
        # Adjoint: D1† * U_hop† * D1†
        D1 = exp_diagonal(U_ctrl[n], Delta_ctrl[n], dt/2, H_int_diag, H_sz_diag)
        U_hop = exp_hop(J_ctrl[n], dt, hop_evals, hop_evecs)

        chi = D1' * chi
        chi = U_hop' * chi
        chi = D1' * chi

        chi_states[n] = copy(chi)
    end

    return chi_states
end

# ============================================================================
# GRAPE implementation with Trotter propagation
# ============================================================================

"""
Compute GRAPE gradients for Trotter propagation.

For fidelity F = |<ψ_target|ψ(T)>|², the gradient is:
∂F/∂u_n = 2 Re{ <ψ_target|ψ(T)>* · <χ_n| ∂U_n/∂u_n |ψ_n> }

where χ_n is backward propagated from ψ_target.
"""
function compute_trotter_gradients(psi_states, chi_states, J_ctrl, U_ctrl, Delta_ctrl, dt,
                                   H_int_diag, H_sz_diag, hop_evals, hop_evecs,
                                   H_hop, H_int, H_sz, overlap)
    num_steps = length(J_ctrl)

    grad_J = zeros(num_steps)
    grad_U = zeros(num_steps)
    grad_Delta = zeros(num_steps)

    for n in 1:num_steps-1
        psi_n = psi_states[n]
        chi_n_plus_1 = chi_states[n+1]

        # Compute gradient contributions using finite difference approximation
        # for the Trotter decomposition

        # For the Trotter step: U_trotter = D1 * U_hop * D1
        # where D1 = exp(-i*(U*H_int + Δ*H_sz)*dt/2), U_hop = exp(-i*J*H_hop*dt)

        D1 = exp_diagonal(U_ctrl[n], Delta_ctrl[n], dt/2, H_int_diag, H_sz_diag)
        U_hop = exp_hop(J_ctrl[n], dt, hop_evals, hop_evecs)

        # Intermediate states in Trotter step
        psi_after_D1 = D1 * psi_n
        psi_after_hop = U_hop * psi_after_D1

        # Gradient w.r.t. J (affects U_hop = exp(-i*J*H_hop*dt))
        # ∂U_hop/∂J = -i*dt*H_hop*U_hop (first order approximation)
        # Full gradient: D1 * (∂U_hop/∂J) * D1 * ψ_n
        dU_hop_dJ_psi = -im * dt * H_hop * (U_hop * psi_after_D1)
        grad_term_J = D1 * dU_hop_dJ_psi
        grad_J[n] = 2 * real(conj(overlap) * dot(chi_n_plus_1, grad_term_J))

        # Gradient w.r.t. U (affects D1 via H_int)
        # D1 = exp(-i*(U*H_int + Δ*H_sz)*dt/2)
        # ∂D1/∂U = -i*(dt/2)*H_int*D1
        # Total: (∂D1/∂U)*U_hop*D1*ψ + D1*U_hop*(∂D1/∂U)*ψ
        dD1_dU_psi = -im * (dt/2) * (H_int * (D1 * psi_n))
        term1 = D1 * U_hop * dD1_dU_psi
        dD1_dU_psi2 = -im * (dt/2) * (H_int * psi_after_hop)
        term2 = D1 * dD1_dU_psi2
        # Wait, need to be more careful. Let me redo this.
        # D1 is diagonal, so D1 * ψ = diag(d) .* ψ where d = exp(-i*α*dt/2) and α = U*H_int_diag + Δ*H_sz_diag
        # ∂d/∂U = -i*(dt/2)*H_int_diag .* d
        # So ∂(D1*ψ)/∂U = (-i*dt/2) * H_int_diag .* (D1*ψ) = (-i*dt/2) * H_int * D1 * ψ

        # For U_trotter = D1 * U_hop * D1:
        # ∂U_trotter/∂U * ψ = (∂D1/∂U)*U_hop*D1*ψ + D1*U_hop*(∂D1/∂U)*ψ
        grad_D1_1 = Diagonal(-im * (dt/2) .* H_int_diag) * psi_after_hop  # second D1
        grad_D1_2 = Diagonal(-im * (dt/2) .* H_int_diag) * psi_after_D1   # first D1
        grad_term_U = D1 * grad_D1_1 + D1 * U_hop * grad_D1_2
        grad_U[n] = 2 * real(conj(overlap) * dot(chi_n_plus_1, grad_term_U))

        # Gradient w.r.t. Delta (affects D1 via H_sz)
        grad_D1_1_delta = Diagonal(-im * (dt/2) .* H_sz_diag) * psi_after_hop
        grad_D1_2_delta = Diagonal(-im * (dt/2) .* H_sz_diag) * psi_after_D1
        grad_term_Delta = D1 * grad_D1_1_delta + D1 * U_hop * grad_D1_2_delta
        grad_Delta[n] = 2 * real(conj(overlap) * dot(chi_n_plus_1, grad_term_Delta))
    end

    return grad_J, grad_U, grad_Delta
end

"""
GRAPE optimization with Trotter propagation using L-BFGS.

Parameters:
- H_hop, H_int, H_sz: Hamiltonian components
- psi0: Initial state
- psi_target: Target state
- T: Total evolution time
- num_steps: Number of time steps
- J0, U0, Delta0: Initial control pulses
- max_iter: Maximum iterations
- tol: Convergence tolerance (infidelity threshold)

Returns optimized control pulses and final fidelity.
"""
function grape_trotter_optimize(H_hop, H_int, H_sz, psi0, psi_target, T, num_steps;
                                 J0=nothing, U0=nothing, Delta0=nothing,
                                 max_iter=1000, tol=1e-6,
                                 verbose=true)

    dt = T / (num_steps - 1)

    # Extract diagonal elements
    H_int_diag = diag(H_int)
    H_sz_diag = diag(H_sz)

    # Precompute H_hop eigendecomposition
    hop_evals, hop_evecs = precompute_hop_eigen(H_hop)

    # Convert H_hop to dense matrix for gradient computation
    H_hop_dense = Matrix(H_hop)

    # Initialize controls
    J_ctrl = isnothing(J0) ? 0.1 * ones(num_steps) : copy(J0)
    U_ctrl = isnothing(U0) ? 0.01 * ones(num_steps) : copy(U0)
    Delta_ctrl = isnothing(Delta0) ? zeros(num_steps) : copy(Delta0)

    # Convert initial/target states
    psi0_vec = Vector{ComplexF64}(psi0)
    psi_target_vec = Vector{ComplexF64}(psi_target)

    # Pack controls into single vector for optimizer: [J; U; Delta]
    function pack_controls(J, U, Delta)
        return vcat(J, U, Delta)
    end

    function unpack_controls(x)
        J = x[1:num_steps]
        U = x[num_steps+1:2*num_steps]
        Delta = x[2*num_steps+1:3*num_steps]
        return J, U, Delta
    end

    # Objective function: minimize infidelity (1 - fidelity)
    function objective(x)
        J, U, Delta = unpack_controls(x)

        psi_final = trotter_propagate(psi0_vec, J, U, Delta, dt,
                                       H_int_diag, H_sz_diag, hop_evals, hop_evecs)

        fidelity = abs2(dot(psi_target_vec, psi_final))
        return 1.0 - fidelity  # Minimize infidelity
    end

    # Gradient function
    function gradient!(g, x)
        J, U, Delta = unpack_controls(x)

        # Forward propagation with storage
        psi_states = trotter_propagate_store(psi0_vec, J, U, Delta, dt,
                                              H_int_diag, H_sz_diag, hop_evals, hop_evecs)
        psi_final = psi_states[end]

        # Compute overlap
        overlap = dot(psi_target_vec, psi_final)

        # Backward propagation
        chi_T = psi_target_vec
        chi_states = trotter_backward_propagate_store(chi_T, J, U, Delta, dt,
                                                       H_int_diag, H_sz_diag, hop_evals, hop_evecs)

        # Compute gradients (of fidelity)
        grad_J, grad_U, grad_Delta = compute_trotter_gradients(
            psi_states, chi_states, J, U, Delta, dt,
            H_int_diag, H_sz_diag, hop_evals, hop_evecs,
            H_hop_dense, H_int, H_sz, overlap)

        # Negate for infidelity gradient (we're minimizing 1-F)
        g[1:num_steps] .= -grad_J
        g[num_steps+1:2*num_steps] .= -grad_U
        g[2*num_steps+1:3*num_steps] .= -grad_Delta
    end

    # Initial packed controls
    x0 = pack_controls(J_ctrl, U_ctrl, Delta_ctrl)

    # Callback for progress
    iter_count = Ref(0)
    function callback(state)
        iter_count[] += 1
        if verbose && (iter_count[] % 10 == 1 || iter_count[] == 1)
            fidelity = 1.0 - state.value
            @printf("Iter %4d: fidelity = %.8f, infidelity = %.2e\n",
                    iter_count[], fidelity, state.value)
        end
        return false
    end

    # Run L-BFGS optimization
    result = optimize(
        objective,
        gradient!,
        x0,
        LBFGS(m=20),  # L-BFGS with memory 20
        Optim.Options(
            iterations=max_iter,
            g_tol=tol * 1e-2,
            f_reltol=tol * 1e-2,
            show_trace=false,
            callback=callback
        )
    )

    # Extract optimized controls
    x_opt = Optim.minimizer(result)
    J_opt, U_opt, Delta_opt = unpack_controls(x_opt)

    # Final fidelity
    final_fidelity = 1.0 - Optim.minimum(result)

    if verbose
        println("\nL-BFGS completed:")
        println("  Iterations: $(Optim.iterations(result))")
        println("  Converged: $(Optim.converged(result))")
    end

    return J_opt, U_opt, Delta_opt, final_fidelity
end

# ============================================================================
# Main execution
# ============================================================================

function main()
    # System parameters
    N = 3  # Number of bosons
    k = 3  # Number of sites

    println("="^60)
    println("GRAPE Optimal Control (Trotter) for Bose-Hubbard Model")
    println("="^60)
    println("N = $N bosons, k = $k sites")
    println("Using 2nd-order Trotter: exp(-iH₁dt/2)·exp(-iH₂dt)·exp(-iH₁dt/2)")
    println("  H₁ = U·H_int + Δ·H_sz (diagonal)")
    println("  H₂ = J·H_hop")

    # Build basis
    basis, ind = make_basis(N, k)
    d = length(basis)
    println("Hilbert space dimension: $d")
    println("First basis state: $(basis[1])")
    println("Last basis state: $(basis[end])")

    # Construct Hamiltonian matrices
    println("\nConstructing Hamiltonian matrices...")
    H_hop = hopping_matrix(basis, ind)
    H_sz = sz_matrix(basis)
    H_int = interaction_matrices(basis)

    # Initial state: |N, 0, 0> (all bosons in first site)
    psi0 = zeros(ComplexF64, d)
    psi0[1] = 1.0
    println("Initial state: |$(basis[1])>")

    # Target state: (|N,0,0> + |0,0,N>)/sqrt(2)  (NOON state)
    psi_target = zeros(ComplexF64, d)
    psi_target[1] = 1/sqrt(2)
    psi_target[end] = 1/sqrt(2)
    println("Target state: (|$(basis[1])> + |$(basis[end])>)/√2")

    # Time parameters
    T = 10
    num_steps = 1001
    dt = T / (num_steps - 1)
    tspan = range(0, T, length=num_steps)
    println("\nTime: T = $T, steps = $num_steps, dt = $dt")

    # Initial guess for controls with random perturbation
    println("\nStarting GRAPE optimization with Trotter propagation...")
    println("-"^60)

    Random.seed!(42)
    t_arr = collect(tspan)
    J0 = 1.0 .+ 0.5 * sin.(2π * t_arr / T) .+ 0.2 * randn(num_steps)
    U0 = 0.1 .+ 0.05 * cos.(2π * t_arr / T) .+ 0.02 * randn(num_steps)
    Delta0 = 0.5 * sin.(4π * t_arr / T) .+ 0.1 * randn(num_steps)

    # Run GRAPE optimization with Trotter (L-BFGS)
    J_opt, U_opt, Delta_opt, final_fidelity = grape_trotter_optimize(
        H_hop, H_int, H_sz, psi0, psi_target, T, num_steps;
        J0=J0, U0=U0, Delta0=Delta0,
        max_iter=500, tol=1e-4,
        verbose=true
    )

    println("-"^60)
    @printf("Final fidelity: %.8f\n", final_fidelity)

    # Save results
    println("\nSaving results...")
    open("J_opt_trotter.txt", "w") do f
        for val in J_opt
            println(f, val)
        end
    end
    open("U_opt_trotter.txt", "w") do f
        for val in U_opt
            println(f, val)
        end
    end
    open("Delta_opt_trotter.txt", "w") do f
        for val in Delta_opt
            println(f, val)
        end
    end
    println("Control pulses saved to J_opt_trotter.txt, U_opt_trotter.txt, Delta_opt_trotter.txt")

    # Final state analysis
    H_int_diag = diag(H_int)
    H_sz_diag = diag(H_sz)
    hop_evals, hop_evecs = precompute_hop_eigen(H_hop)
    dt = T / (num_steps - 1)

    psi_final = trotter_propagate(Vector{ComplexF64}(psi0), J_opt, U_opt, Delta_opt, dt,
                                   H_int_diag, H_sz_diag, hop_evals, hop_evecs)

    println("\nFinal state analysis:")
    println("  |<ψ_target|ψ_final>|² = $(abs2(dot(psi_target, psi_final)))")
    println("  |ψ_final[1]|² = $(abs2(psi_final[1]))")
    println("  |ψ_final[end]|² = $(abs2(psi_final[end]))")

    return J_opt, U_opt, Delta_opt, final_fidelity
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
