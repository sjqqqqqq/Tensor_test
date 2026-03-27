"""
GRAPE (Gradient Ascent Pulse Engineering) for MPS quantum optimal control - OPTIMIZED VERSION.

Performance improvements:
1. Precompute static gradient operators once at initialization
2. Cache H1/H2 base operators to avoid repeated construction
3. Optimized expectation value computation with incremental environment building
4. Reduced MPS copies by reusing states where possible
5. Batch gate construction with caching
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Random
using Printf
using Optim
using Statistics

# ============================================================================
# Configuration
# ============================================================================

struct GRAPEConfig
    n_sites::Int
    n_particles::Int
    n_steps::Int
    dt::Float64
    cutoff::Float64
    max_iterations::Int
    tolerance::Float64
end

# ============================================================================
# Precomputed operators cache
# ============================================================================

struct OperatorCache
    # Base operators for H1 (indexed by site)
    N_op::Vector{ITensor}           # N operator at each site
    NNmN_op::Vector{ITensor}        # N*N - N operator at each site
    site_offset::Vector{Float64}    # (i - center) for each site

    # Base operators for H2 (indexed by bond j -> j+1)
    AdagA_op::Vector{ITensor}       # Adag_j * A_{j+1}
    AAdagop::Vector{ITensor}        # A_j * Adag_{j+1}

    # Precomputed gradient operators
    grad_H1_U::Vector{ITensor}      # ∂H1/∂U = N*N - N
    grad_H1_Δ::Vector{ITensor}      # ∂H1/∂Δ = (i-center)*N
    grad_H2_J::Vector{ITensor}      # ∂H2/∂J = Adag_j*A_{j+1} + A_j*Adag_{j+1}
end

"""Build operator cache for fast repeated access."""
function build_operator_cache(s, n_sites::Int)
    center = (n_sites + 1) / 2

    N_op = [op("N", s[j]) for j in 1:n_sites]
    NNmN_op = [op("N * N", s[j]) - op("N", s[j]) for j in 1:n_sites]
    site_offset = [(j - center) for j in 1:n_sites]

    AdagA_op = [op("Adag", s[j]) * op("A", s[j+1]) for j in 1:(n_sites-1)]
    AAdagop = [op("A", s[j]) * op("Adag", s[j+1]) for j in 1:(n_sites-1)]

    grad_H1_U = NNmN_op  # Same as NNmN
    grad_H1_Δ = [site_offset[j] * N_op[j] for j in 1:n_sites]
    grad_H2_J = [AdagA_op[j] + AAdagop[j] for j in 1:(n_sites-1)]

    return OperatorCache(N_op, NNmN_op, site_offset, AdagA_op, AAdagop,
                         grad_H1_U, grad_H1_Δ, grad_H2_J)
end

# ============================================================================
# Hamiltonian construction (using cache)
# ============================================================================

"""Build on-site Hamiltonian using cached operators."""
function build_H1_cached(site_idx::Int, U::Float64, Δ::Float64, cache::OperatorCache)
    return U * cache.NNmN_op[site_idx] + Δ * cache.grad_H1_Δ[site_idx]
end

"""Build hopping Hamiltonian using cached operators."""
function build_H2_cached(site_idx::Int, J::Float64, cache::OperatorCache)
    return J * cache.grad_H2_J[site_idx]
end

# ============================================================================
# Gate construction (optimized)
# ============================================================================

"""Build H1 gates for all sites using cached operators."""
function make_H1_gates_cached(U::Float64, Δ::Float64, dt_factor::Float64,
                               cache::OperatorCache, n_sites::Int)
    return [exp(-im * dt_factor * build_H1_cached(j, U, Δ, cache)) for j in 1:n_sites]
end

"""Build H2 gates for all bonds using cached operators."""
function make_H2_gates_cached(J::Float64, dt::Float64, cache::OperatorCache, n_sites::Int)
    return [exp(-im * dt * build_H2_cached(j, J, cache)) for j in 1:(n_sites-1)]
end

"""Build adjoint H1 gates (for backward propagation)."""
function make_H1_gates_adj_cached(U::Float64, Δ::Float64, dt_factor::Float64,
                                   cache::OperatorCache, n_sites::Int)
    return [exp(+im * dt_factor * build_H1_cached(j, U, Δ, cache)) for j in n_sites:-1:1]
end

"""Build adjoint H2 gates (for backward propagation)."""
function make_H2_gates_adj_cached(J::Float64, dt::Float64, cache::OperatorCache, n_sites::Int)
    return [exp(+im * dt * build_H2_cached(j, J, cache)) for j in (n_sites-1):-1:1]
end

# ============================================================================
# State initialization
# ============================================================================

"""Create initial state: |N,0,...,0⟩"""
function make_initial_state(s, config::GRAPEConfig)
    state = vcat(["$(config.n_particles)"], fill("0", config.n_sites - 1))
    return MPS(s, state)
end

"""Create NOON target state: (|N,0,...,0⟩ + |0,...,0,N⟩)/√2"""
function make_noon_state(s, config::GRAPEConfig)
    n = config.n_particles
    state1 = vcat(["$n"], fill("0", config.n_sites - 1))
    state2 = vcat(fill("0", config.n_sites - 1), ["$n"])

    psi1 = MPS(s, state1)
    psi2 = MPS(s, state2)

    noon = add(psi1, psi2; cutoff=config.cutoff)
    noon ./= sqrt(2)
    normalize!(noon)
    return noon
end

# ============================================================================
# MPS propagation
# ============================================================================

"""Forward propagate without storing intermediate states."""
function forward_propagate(psi0::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                           Δ_vec::Vector{Float64}, cache::OperatorCache, config::GRAPEConfig)
    psi = copy(psi0)
    n_sites, dt, cutoff = config.n_sites, config.dt, config.cutoff

    for n in 1:config.n_steps-1
        # Build gates for this time step
        gates_H1 = make_H1_gates_cached(U_vec[n], Δ_vec[n], dt/2, cache, n_sites)
        gates_H2 = make_H2_gates_cached(J_vec[n], dt, cache, n_sites)

        # Apply: H1(dt/2) -> H2(dt) -> H1(dt/2)
        psi = apply(gates_H1, psi; cutoff=cutoff)
        psi = apply(gates_H2, psi; cutoff=cutoff)
        psi = apply(gates_H1, psi; cutoff=cutoff)
        normalize!(psi)
    end
    return psi
end

"""Compute fidelity F = |⟨ψ_target|ψ(T)⟩|²"""
function compute_fidelity(psi0::MPS, psi_target::MPS, J_vec::Vector{Float64},
                          U_vec::Vector{Float64}, Δ_vec::Vector{Float64},
                          cache::OperatorCache, config::GRAPEConfig)
    psi_final = forward_propagate(psi0, J_vec, U_vec, Δ_vec, cache, config)
    return abs2(inner(psi_target, psi_final))
end

# ============================================================================
# Optimized expectation values
# ============================================================================

"""Compute ⟨χ|H|ψ⟩ for single-site operator H at site j.
   Assumes chi and psi are already orthogonalized to site j."""
function expect_single_site_fast(chi::MPS, psi::MPS, H_op::ITensor, site_j::Int,
                                  L_env::ITensor, R_env::ITensor)
    # Apply operator to psi at site j
    H_psi_j = noprime(H_op * psi[site_j])

    # Contract: L_env * dag(chi[j]) * H_psi_j * R_env
    result = L_env * dag(chi[site_j]) * H_psi_j * R_env
    return scalar(result)
end

"""Compute ⟨χ|H|ψ⟩ for two-site operator H at sites j, j+1.
   chi and psi should be orthogonalized near site j."""
function expect_two_site_fast(chi::MPS, psi::MPS, H_op::ITensor, site_j::Int,
                               L_env::ITensor, R_env::ITensor)
    # Local two-site contraction
    psi_local = psi[site_j] * psi[site_j + 1]
    H_psi = noprime(H_op * psi_local)
    chi_local = dag(chi[site_j]) * dag(chi[site_j + 1])

    return scalar(L_env * chi_local * H_psi * R_env)
end

"""Build left environment from site 1 to site j-1."""
function build_left_env(chi::MPS, psi::MPS, j::Int)
    if j == 1
        return ITensor(1.0)
    end
    L = ITensor(1.0)
    for i in 1:(j-1)
        L = L * dag(chi[i]) * psi[i]
    end
    return L
end

"""Build right environment from site n down to site j+1."""
function build_right_env(chi::MPS, psi::MPS, j::Int, n::Int)
    if j == n
        return ITensor(1.0)
    end
    R = ITensor(1.0)
    for i in n:-1:(j+1)
        R = R * dag(chi[i]) * psi[i]
    end
    return R
end

# ============================================================================
# Analytical gradient computation (OPTIMIZED)
# ============================================================================

"""
Forward propagation storing states at key points.
Optimized: builds gates once per time step, reduces copies.
"""
function forward_propagate_detailed_fast(psi0::MPS, J_vec::Vector{Float64},
                                          U_vec::Vector{Float64}, Δ_vec::Vector{Float64},
                                          cache::OperatorCache, config::GRAPEConfig)
    n_steps, n_sites, dt, cutoff = config.n_steps, config.n_sites, config.dt, config.cutoff

    # Storage for gradient computation
    psi_after_H1_first = Vector{MPS}(undef, n_steps)   # After first H1(dt/2)
    psi_after_H2 = Vector{MPS}(undef, n_steps)         # After H2(dt)
    psi_after_H1_second = Vector{MPS}(undef, n_steps)  # After second H1(dt/2) = start of next step

    psi = copy(psi0)

    for n in 1:n_steps-1
        # Build all gates for this time step once
        gates_H1 = make_H1_gates_cached(U_vec[n], Δ_vec[n], dt/2, cache, n_sites)
        gates_H2 = make_H2_gates_cached(J_vec[n], dt, cache, n_sites)

        # First H1 (dt/2)
        psi = apply(gates_H1, psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H1_first[n] = copy(psi)

        # H2 (dt)
        psi = apply(gates_H2, psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H2[n] = copy(psi)

        # Second H1 (dt/2)
        psi = apply(gates_H1, psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H1_second[n] = copy(psi)
    end

    # Final state placeholders
    psi_after_H1_first[n_steps] = copy(psi)
    psi_after_H2[n_steps] = copy(psi)
    psi_after_H1_second[n_steps] = copy(psi)

    return psi_after_H1_first, psi_after_H2, psi_after_H1_second
end

"""
Backward propagation storing states at key points.
Optimized: builds gates once per time step.
"""
function backward_propagate_detailed_fast(chi_T::MPS, J_vec::Vector{Float64},
                                           U_vec::Vector{Float64}, Δ_vec::Vector{Float64},
                                           cache::OperatorCache, config::GRAPEConfig)
    n_steps, n_sites, dt, cutoff = config.n_steps, config.n_sites, config.dt, config.cutoff

    # chi_before_H2[n] = chi state to use for J gradient at step n
    # chi_before_H1_second[n] = chi state to use for U,Δ gradient (second H1) at step n
    chi_before_H2 = Vector{MPS}(undef, n_steps)
    chi_before_H1_second = Vector{MPS}(undef, n_steps)

    chi = copy(chi_T)
    chi_before_H2[n_steps] = copy(chi)
    chi_before_H1_second[n_steps] = copy(chi)

    for n in n_steps-1:-1:1
        # Build adjoint gates for this time step
        gates_H1_adj = make_H1_gates_adj_cached(U_vec[n], Δ_vec[n], dt/2, cache, n_sites)
        gates_H2_adj = make_H2_gates_adj_cached(J_vec[n], dt, cache, n_sites)

        # Backward through second H1
        chi = apply(gates_H1_adj, chi; cutoff=cutoff)
        normalize!(chi)
        chi_before_H1_second[n] = copy(chi)  # Use for second H1 gradient

        # Backward through H2
        chi = apply(gates_H2_adj, chi; cutoff=cutoff)
        normalize!(chi)
        chi_before_H2[n] = copy(chi)  # Use for J gradient

        # Backward through first H1
        chi = apply(gates_H1_adj, chi; cutoff=cutoff)
        normalize!(chi)
    end

    return chi_before_H2, chi_before_H1_second
end

"""
Compute analytical GRAPE gradients - OPTIMIZED VERSION.

Key optimizations:
1. Uses precomputed gradient operators from cache
2. Computes environments incrementally where possible
3. Orthogonalizes MPS only once per state pair
"""
function compute_gradients_analytical_fast(psi0::MPS, psi_target::MPS,
                                            J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                            Δ_vec::Vector{Float64}, cache::OperatorCache,
                                            config::GRAPEConfig)
    n_steps, n_sites, dt = config.n_steps, config.n_sites, config.dt

    grad_J = zeros(n_steps)
    grad_U = zeros(n_steps)
    grad_Δ = zeros(n_steps)

    # Forward and backward propagation
    psi_after_H1, psi_after_H2, psi_after_full = forward_propagate_detailed_fast(
        psi0, J_vec, U_vec, Δ_vec, cache, config)

    chi_before_H2, chi_before_H1_second = backward_propagate_detailed_fast(
        psi_target, J_vec, U_vec, Δ_vec, cache, config)

    # Compute overlap for gradient formula
    psi_final = psi_after_full[n_steps-1]
    overlap = inner(psi_target, psi_final)
    F = abs2(overlap)

    for n in 1:n_steps-1
        # ===== J gradient (from H2 gate) =====
        chi_J = chi_before_H1_second[n]  # chi before second H1 = chi after H2 backward
        psi_J = psi_after_H2[n]

        # Orthogonalize once for J gradient computation
        chi_orth = orthogonalize(chi_J, 1)
        psi_orth = orthogonalize(psi_J, 1)

        # Build environments incrementally for two-site operators
        L_env = ITensor(1.0)
        for j in 1:(n_sites-1)
            # Right environment for sites j, j+1
            R_env = build_right_env(chi_orth, psi_orth, j+1, n_sites)

            expect_val = expect_two_site_fast(chi_orth, psi_orth, cache.grad_H2_J[j], j, L_env, R_env)
            grad_J[n] += 2.0 * real(conj(overlap) * (-im * dt) * expect_val)

            # Update left environment for next iteration
            L_env = L_env * dag(chi_orth[j]) * psi_orth[j]
        end

        # ===== U and Δ gradients (from H1 gates) =====

        # First H1 gradient: use chi after backward through H2, psi after first H1
        chi_first = chi_before_H2[n]
        psi_first = psi_after_H1[n]

        chi_orth = orthogonalize(chi_first, 1)
        psi_orth = orthogonalize(psi_first, 1)

        L_env = ITensor(1.0)
        for j in 1:n_sites
            R_env = build_right_env(chi_orth, psi_orth, j, n_sites)

            expect_U = expect_single_site_fast(chi_orth, psi_orth, cache.grad_H1_U[j], j, L_env, R_env)
            expect_Δ = expect_single_site_fast(chi_orth, psi_orth, cache.grad_H1_Δ[j], j, L_env, R_env)

            grad_U[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_U)
            grad_Δ[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_Δ)

            L_env = L_env * dag(chi_orth[j]) * psi_orth[j]
        end

        # Second H1 gradient: use chi before second H1, psi after H2
        chi_second = chi_before_H1_second[n]
        psi_second = psi_after_H2[n]

        chi_orth = orthogonalize(chi_second, 1)
        psi_orth = orthogonalize(psi_second, 1)

        L_env = ITensor(1.0)
        for j in 1:n_sites
            R_env = build_right_env(chi_orth, psi_orth, j, n_sites)

            expect_U = expect_single_site_fast(chi_orth, psi_orth, cache.grad_H1_U[j], j, L_env, R_env)
            expect_Δ = expect_single_site_fast(chi_orth, psi_orth, cache.grad_H1_Δ[j], j, L_env, R_env)

            grad_U[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_U)
            grad_Δ[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_Δ)

            L_env = L_env * dag(chi_orth[j]) * psi_orth[j]
        end
    end

    return grad_J, grad_U, grad_Δ, F
end

"""Compute numerical gradients using finite differences."""
function compute_gradients_numerical(psi0::MPS, psi_target::MPS,
                                      J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                      Δ_vec::Vector{Float64}, cache::OperatorCache,
                                      config::GRAPEConfig; ε::Float64=1e-6)
    n_steps = config.n_steps
    grad_J, grad_U, grad_Δ = zeros(n_steps), zeros(n_steps), zeros(n_steps)

    F0 = compute_fidelity(psi0, psi_target, J_vec, U_vec, Δ_vec, cache, config)

    for n in 1:n_steps-1
        # J gradient
        J_plus = copy(J_vec); J_plus[n] += ε
        F_plus = compute_fidelity(psi0, psi_target, J_plus, U_vec, Δ_vec, cache, config)
        grad_J[n] = (F_plus - F0) / ε

        # U gradient
        U_plus = copy(U_vec); U_plus[n] += ε
        F_plus = compute_fidelity(psi0, psi_target, J_vec, U_plus, Δ_vec, cache, config)
        grad_U[n] = (F_plus - F0) / ε

        # Δ gradient
        Δ_plus = copy(Δ_vec); Δ_plus[n] += ε
        F_plus = compute_fidelity(psi0, psi_target, J_vec, U_vec, Δ_plus, cache, config)
        grad_Δ[n] = (F_plus - F0) / ε
    end

    return grad_J, grad_U, grad_Δ, F0
end

# ============================================================================
# Optimization
# ============================================================================

"""GRAPE optimization using gradient descent with momentum."""
function grape_optimize(s, cache::OperatorCache, config::GRAPEConfig;
                        J0=nothing, U0=nothing, Δ0=nothing,
                        learning_rate=0.1, momentum=0.9,
                        use_analytical=true, verbose=true)
    n_steps = config.n_steps

    J = isnothing(J0) ? 0.5 * ones(n_steps) : copy(J0)
    U = isnothing(U0) ? 0.1 * ones(n_steps) : copy(U0)
    Δ = isnothing(Δ0) ? zeros(n_steps) : copy(Δ0)

    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    best_fidelity = 0.0
    best_J, best_U, best_Δ = copy(J), copy(U), copy(Δ)
    v_J, v_U, v_Δ = zeros(n_steps), zeros(n_steps), zeros(n_steps)
    lr, prev_fidelity = learning_rate, 0.0

    verbose && println("Using $(use_analytical ? "analytical" : "numerical") gradients")

    for iter in 1:config.max_iterations
        grad_J, grad_U, grad_Δ, fidelity = use_analytical ?
            compute_gradients_analytical_fast(psi0, psi_target, J, U, Δ, cache, config) :
            compute_gradients_numerical(psi0, psi_target, J, U, Δ, cache, config)

        if fidelity > best_fidelity
            best_fidelity = fidelity
            best_J, best_U, best_Δ = copy(J), copy(U), copy(Δ)
        end

        verbose && (iter == 1 || iter % 10 == 0) &&
            @printf("Iter %3d: fidelity = %.6f (best = %.6f)\n", iter, fidelity, best_fidelity)

        best_fidelity > 1.0 - config.tolerance && (verbose && println("Converged!"); break)

        # Adaptive learning rate
        if iter > 1
            lr = fidelity < prev_fidelity - 0.05 ? lr * 0.7 :
                 fidelity > prev_fidelity ? min(lr * 1.02, learning_rate) : lr
        end
        prev_fidelity = fidelity

        # Momentum update
        v_J .= momentum .* v_J .+ lr .* grad_J
        v_U .= momentum .* v_U .+ lr .* grad_U
        v_Δ .= momentum .* v_Δ .+ lr .* grad_Δ

        J .+= v_J; J .= max.(J, 0.01)
        U .+= v_U
        Δ .+= v_Δ
    end

    verbose && println("Best fidelity: $best_fidelity")
    return best_J, best_U, best_Δ, best_fidelity
end

"""GRAPE optimization using L-BFGS."""
function grape_optimize_lbfgs(s, cache::OperatorCache, config::GRAPEConfig;
                              J0=nothing, U0=nothing, Δ0=nothing,
                              use_analytical=true, verbose=true)
    n_steps = config.n_steps

    J = isnothing(J0) ? 0.1 * ones(n_steps) : copy(J0)
    U = isnothing(U0) ? 0.01 * ones(n_steps) : copy(U0)
    Δ = isnothing(Δ0) ? zeros(n_steps) : copy(Δ0)

    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    pack(J, U, Δ) = vcat(J, U, Δ)
    unpack(x) = (x[1:n_steps], x[n_steps+1:2n_steps], x[2n_steps+1:3n_steps])

    objective(x) = 1.0 - compute_fidelity(psi0, psi_target, unpack(x)..., cache, config)

    function gradient!(g, x)
        grad_J, grad_U, grad_Δ, _ = use_analytical ?
            compute_gradients_analytical_fast(psi0, psi_target, unpack(x)..., cache, config) :
            compute_gradients_numerical(psi0, psi_target, unpack(x)..., cache, config)
        g[1:n_steps] .= -grad_J
        g[n_steps+1:2n_steps] .= -grad_U
        g[2n_steps+1:3n_steps] .= -grad_Δ
    end

    iter_count = Ref(0)
    callback(state) = (iter_count[] += 1;
        verbose && iter_count[] % 10 == 1 && @printf("Iter %4d: fidelity = %.8f\n",
            iter_count[], 1.0 - state.value);
        false)

    result = optimize(objective, gradient!, pack(J, U, Δ), LBFGS(m=20),
        Optim.Options(iterations=config.max_iterations, g_tol=config.tolerance*1e-2,
                      f_reltol=config.tolerance*1e-2, callback=callback))

    J_opt, U_opt, Δ_opt = unpack(Optim.minimizer(result))
    final_fidelity = 1.0 - Optim.minimum(result)

    verbose && println("\nL-BFGS: $(Optim.iterations(result)) iterations, converged=$(Optim.converged(result))")
    return J_opt, U_opt, Δ_opt, final_fidelity
end

# ============================================================================
# Analysis
# ============================================================================

"""Analyze if state is a NOON state."""
function analyze_noon_state(psi::MPS, s, config::GRAPEConfig)
    n = config.n_particles

    psi_left = MPS(s, vcat(["$n"], fill("0", config.n_sites - 1)))
    psi_right = MPS(s, vcat(fill("0", config.n_sites - 1), ["$n"]))
    target = make_noon_state(s, config)

    return (prob_left=abs2(inner(psi_left, psi)),
            prob_right=abs2(inner(psi_right, psi)),
            fidelity=abs2(inner(target, psi)))
end

"""Test analytical vs numerical gradients."""
function test_gradients(s, cache::OperatorCache, config::GRAPEConfig; verbose=true)
    Random.seed!(123)
    n_steps = config.n_steps

    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    J = 0.5 .+ 0.2 * randn(n_steps)
    U = 0.1 .+ 0.05 * randn(n_steps)
    Δ = 0.2 * randn(n_steps)

    grad_ana = compute_gradients_analytical_fast(psi0, psi_target, J, U, Δ, cache, config)
    grad_num = compute_gradients_numerical(psi0, psi_target, J, U, Δ, cache, config)

    idx = 1:n_steps-1
    correlations = [cor(grad_ana[i][idx], grad_num[i][idx]) for i in 1:3]

    if verbose
        println("Gradient test (dt = $(config.dt)):")
        @printf("  Correlations: J=%.4f, U=%.4f, Δ=%.4f\n", correlations...)
        @printf("  Fidelity: analytical=%.6f, numerical=%.6f\n", grad_ana[4], grad_num[4])
    end

    return correlations
end

# ============================================================================
# Main
# ============================================================================

function main()
    N, k, T = 3, 3, 10.0
    num_steps = 101
    dt = T / (num_steps - 1)

    println("="^60)
    println("GRAPE Optimal Control (MPS) - OPTIMIZED VERSION")
    println("="^60)
    println("N=$N bosons, k=$k sites, T=$T, dt=$(@sprintf("%.3f", dt))")

    config = GRAPEConfig(k, N, num_steps, dt, 1e-8, 300, 1e-3)
    s = siteinds("Boson", k; dim=N+1, conserve_qns=true)

    # Build operator cache ONCE
    println("\nBuilding operator cache...")
    cache = build_operator_cache(s, k)
    println("Cache built.")

    # Test gradients
    println("\nGradient verification:")
    test_gradients(s, cache, config)

    # Run optimization
    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    println("\n" * "="^60)
    println("Optimization")
    println("="^60)
    @printf("Initial fidelity: %.6f\n\n", abs2(inner(psi_target, psi0)))

    Random.seed!(42)
    t = range(0, T, length=num_steps)
    J0 = 1.0 .+ 0.3 * sin.(2π * t / T) .+ 0.1 * randn(num_steps)
    U0 = 0.1 .+ 0.05 * cos.(2π * t / T) .+ 0.02 * randn(num_steps)
    Δ0 = 0.3 * sin.(4π * t / T) .+ 0.05 * randn(num_steps)

    J_opt, U_opt, Δ_opt, fidelity = grape_optimize(s, cache, config;
                                                    J0=J0, U0=U0, Δ0=Δ0,
                                                    use_analytical=true)

    println("\n" * "-"^60)
    @printf("Final fidelity: %.6f\n", fidelity)

    psi_final = forward_propagate(psi0, J_opt, U_opt, Δ_opt, cache, config)
    result = analyze_noon_state(psi_final, s, config)

    println("\nFinal state analysis:")
    @printf("  P(|%d,0,0⟩) = %.6f\n", N, result.prob_left)
    @printf("  P(|0,0,%d⟩) = %.6f\n", N, result.prob_right)

    # Save results
    for (name, data) in [("J_opt_mps.txt", J_opt), ("U_opt_mps.txt", U_opt), ("Delta_opt_mps.txt", Δ_opt)]
        open(name, "w") do f; foreach(v -> println(f, v), data); end
    end
    println("\nControl pulses saved.")

    return J_opt, U_opt, Δ_opt, fidelity
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
