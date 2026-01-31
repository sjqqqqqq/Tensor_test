"""
GRAPE for MPS - Version 2 with aggressive optimizations.

Key optimizations:
1. Precompute ALL gates for all time steps at start of gradient computation
2. Fused Trotter step application
3. Minimal MPS copies using careful state management
4. Single orthogonalization sweep for expectation values
5. In-place operations where possible
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
# Precomputed operators
# ============================================================================

struct OperatorCache
    n_sites::Int
    N_op::Vector{ITensor}
    NNmN_op::Vector{ITensor}
    site_offset::Vector{Float64}
    grad_H2_J::Vector{ITensor}
end

function build_operator_cache(s, n_sites::Int)
    center = (n_sites + 1) / 2
    N_op = [op("N", s[j]) for j in 1:n_sites]
    NNmN_op = [op("N * N", s[j]) - op("N", s[j]) for j in 1:n_sites]
    site_offset = [(j - center) for j in 1:n_sites]
    grad_H2_J = [op("Adag", s[j]) * op("A", s[j+1]) + op("A", s[j]) * op("Adag", s[j+1])
                 for j in 1:(n_sites-1)]
    return OperatorCache(n_sites, N_op, NNmN_op, site_offset, grad_H2_J)
end

# ============================================================================
# Gate construction
# ============================================================================

"""Build all gates for a single Trotter step. Returns (H1_gates, H2_gates)."""
function make_trotter_gates_cached(J::Float64, U::Float64, Δ::Float64, dt::Float64,
                                    cache::OperatorCache)
    n = cache.n_sites
    # H1 gates (used twice with dt/2)
    H1_gates = [exp(-im * dt/2 * (U * cache.NNmN_op[j] + Δ * cache.site_offset[j] * cache.N_op[j]))
                for j in 1:n]
    # H2 gates
    H2_gates = [exp(-im * dt * J * cache.grad_H2_J[j]) for j in 1:(n-1)]
    return H1_gates, H2_gates
end

"""Build adjoint gates for backward propagation."""
function make_trotter_gates_adj_cached(J::Float64, U::Float64, Δ::Float64, dt::Float64,
                                        cache::OperatorCache)
    n = cache.n_sites
    H1_gates_adj = [exp(+im * dt/2 * (U * cache.NNmN_op[j] + Δ * cache.site_offset[j] * cache.N_op[j]))
                    for j in n:-1:1]
    H2_gates_adj = [exp(+im * dt * J * cache.grad_H2_J[j]) for j in (n-1):-1:1]
    return H1_gates_adj, H2_gates_adj
end

# ============================================================================
# State initialization
# ============================================================================

function make_initial_state(s, config::GRAPEConfig)
    state = vcat(["$(config.n_particles)"], fill("0", config.n_sites - 1))
    return MPS(s, state)
end

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
# MPS propagation - simplified
# ============================================================================

"""Apply a Trotter step: H1(dt/2) -> H2(dt) -> H1(dt/2)."""
function apply_trotter_step!(psi::MPS, H1_gates, H2_gates, cutoff::Float64)
    psi = apply(H1_gates, psi; cutoff=cutoff)
    psi = apply(H2_gates, psi; cutoff=cutoff)
    psi = apply(H1_gates, psi; cutoff=cutoff)
    normalize!(psi)
    return psi
end

"""Forward propagate to final state."""
function forward_propagate(psi0::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                           Δ_vec::Vector{Float64}, cache::OperatorCache, config::GRAPEConfig)
    psi = copy(psi0)
    for n in 1:config.n_steps-1
        H1, H2 = make_trotter_gates_cached(J_vec[n], U_vec[n], Δ_vec[n], config.dt, cache)
        psi = apply_trotter_step!(psi, H1, H2, config.cutoff)
    end
    return psi
end

"""Compute fidelity."""
function compute_fidelity(psi0::MPS, psi_target::MPS, J_vec::Vector{Float64},
                          U_vec::Vector{Float64}, Δ_vec::Vector{Float64},
                          cache::OperatorCache, config::GRAPEConfig)
    return abs2(inner(psi_target, forward_propagate(psi0, J_vec, U_vec, Δ_vec, cache, config)))
end

# ============================================================================
# Efficient expectation values via environment caching
# ============================================================================

"""
Build left environments L[j] = ⟨χ[1..j-1]|ψ[1..j-1]⟩ for j = 1..n+1.
L[1] = 1, L[j] includes contraction up to site j-1.
"""
function build_left_environments(chi::MPS, psi::MPS)
    n = length(psi)
    L = Vector{ITensor}(undef, n + 1)
    L[1] = ITensor(1.0)
    for j in 1:n
        L[j + 1] = L[j] * dag(chi[j]) * psi[j]
    end
    return L
end

"""
Build right environments R[j] = ⟨χ[j+1..n]|ψ[j+1..n]⟩ for j = 0..n.
R[n] = 1, R[j] includes contraction from site j+1 to n.
"""
function build_right_environments(chi::MPS, psi::MPS)
    n = length(psi)
    R = Vector{ITensor}(undef, n + 1)
    R[n + 1] = ITensor(1.0)
    for j in n:-1:1
        R[j] = R[j + 1] * dag(chi[j]) * psi[j]
    end
    return R
end

"""Compute ⟨χ|O_j|ψ⟩ using precomputed environments."""
function expect_single_site_with_env(chi::MPS, psi::MPS, O::ITensor, j::Int,
                                      L::Vector{ITensor}, R::Vector{ITensor})
    return scalar(L[j] * dag(chi[j]) * noprime(O * psi[j]) * R[j + 1])
end

"""Compute ⟨χ|O_{j,j+1}|ψ⟩ using precomputed environments."""
function expect_two_site_with_env(chi::MPS, psi::MPS, O::ITensor, j::Int,
                                   L::Vector{ITensor}, R::Vector{ITensor})
    psi_local = psi[j] * psi[j + 1]
    O_psi = noprime(O * psi_local)
    chi_local = dag(chi[j]) * dag(chi[j + 1])
    return scalar(L[j] * chi_local * O_psi * R[j + 2])
end

# ============================================================================
# Gradient computation - optimized with precomputed gates and environments
# ============================================================================

"""
Compute analytical gradients with full gate and environment precomputation.
"""
function compute_gradients_analytical(psi0::MPS, psi_target::MPS,
                                       J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                       Δ_vec::Vector{Float64}, cache::OperatorCache,
                                       config::GRAPEConfig)
    n_steps, n_sites, dt, cutoff = config.n_steps, config.n_sites, config.dt, config.cutoff

    grad_J = zeros(n_steps)
    grad_U = zeros(n_steps)
    grad_Δ = zeros(n_steps)

    # ===== Precompute ALL gates for all time steps =====
    all_H1_gates = Vector{Vector{ITensor}}(undef, n_steps)
    all_H2_gates = Vector{Vector{ITensor}}(undef, n_steps)
    all_H1_gates_adj = Vector{Vector{ITensor}}(undef, n_steps)
    all_H2_gates_adj = Vector{Vector{ITensor}}(undef, n_steps)

    for n in 1:n_steps-1
        all_H1_gates[n], all_H2_gates[n] = make_trotter_gates_cached(
            J_vec[n], U_vec[n], Δ_vec[n], dt, cache)
        all_H1_gates_adj[n], all_H2_gates_adj[n] = make_trotter_gates_adj_cached(
            J_vec[n], U_vec[n], Δ_vec[n], dt, cache)
    end

    # ===== Forward propagation with state storage =====
    # Store states at 3 points per time step:
    # - after first H1: for first H1 gradient
    # - after H2: for J gradient and second H1 gradient
    psi_after_H1_first = Vector{MPS}(undef, n_steps)
    psi_after_H2 = Vector{MPS}(undef, n_steps)

    psi = copy(psi0)
    for n in 1:n_steps-1
        # First H1
        psi = apply(all_H1_gates[n], psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H1_first[n] = copy(psi)

        # H2
        psi = apply(all_H2_gates[n], psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H2[n] = copy(psi)

        # Second H1
        psi = apply(all_H1_gates[n], psi; cutoff=cutoff)
        normalize!(psi)
    end
    psi_final = psi

    # ===== Backward propagation with state storage =====
    chi_before_H1_second = Vector{MPS}(undef, n_steps)  # For second H1 and J gradients
    chi_before_H2 = Vector{MPS}(undef, n_steps)         # For first H1 gradient

    chi = copy(psi_target)
    for n in n_steps-1:-1:1
        # Backward through second H1
        chi = apply(all_H1_gates_adj[n], chi; cutoff=cutoff)
        normalize!(chi)
        chi_before_H1_second[n] = copy(chi)

        # Backward through H2
        chi = apply(all_H2_gates_adj[n], chi; cutoff=cutoff)
        normalize!(chi)
        chi_before_H2[n] = copy(chi)

        # Backward through first H1
        chi = apply(all_H1_gates_adj[n], chi; cutoff=cutoff)
        normalize!(chi)
    end

    # Overlap for gradient formula
    overlap = inner(psi_target, psi_final)
    F = abs2(overlap)

    # ===== Compute gradients with precomputed environments =====
    for n in 1:n_steps-1
        # Precompute gradient operators for this time step
        grad_H1_U = cache.NNmN_op
        grad_H1_Δ = [cache.site_offset[j] * cache.N_op[j] for j in 1:n_sites]

        # --- J gradient (from H2) ---
        chi_J = orthogonalize(chi_before_H1_second[n], 1)
        psi_J = orthogonalize(psi_after_H2[n], 1)
        L = build_left_environments(chi_J, psi_J)
        R = build_right_environments(chi_J, psi_J)

        for j in 1:(n_sites-1)
            expect_val = expect_two_site_with_env(chi_J, psi_J, cache.grad_H2_J[j], j, L, R)
            grad_J[n] += 2.0 * real(conj(overlap) * (-im * dt) * expect_val)
        end

        # --- First H1 gradient (dt/2) ---
        chi_1 = orthogonalize(chi_before_H2[n], 1)
        psi_1 = orthogonalize(psi_after_H1_first[n], 1)
        L = build_left_environments(chi_1, psi_1)
        R = build_right_environments(chi_1, psi_1)

        for j in 1:n_sites
            expect_U = expect_single_site_with_env(chi_1, psi_1, grad_H1_U[j], j, L, R)
            expect_Δ = expect_single_site_with_env(chi_1, psi_1, grad_H1_Δ[j], j, L, R)
            grad_U[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_U)
            grad_Δ[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_Δ)
        end

        # --- Second H1 gradient (dt/2) ---
        # Use already orthogonalized chi_J and re-orthogonalized psi_after_H2
        chi_2 = chi_J  # Already orthogonalized
        psi_2 = psi_J  # Already orthogonalized
        # Reuse L and R from J gradient section? No, different pair. Rebuild.
        L = build_left_environments(chi_2, psi_2)
        R = build_right_environments(chi_2, psi_2)

        for j in 1:n_sites
            expect_U = expect_single_site_with_env(chi_2, psi_2, grad_H1_U[j], j, L, R)
            expect_Δ = expect_single_site_with_env(chi_2, psi_2, grad_H1_Δ[j], j, L, R)
            grad_U[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_U)
            grad_Δ[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_Δ)
        end
    end

    return grad_J, grad_U, grad_Δ, F
end

"""Compute numerical gradients."""
function compute_gradients_numerical(psi0::MPS, psi_target::MPS,
                                      J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                      Δ_vec::Vector{Float64}, cache::OperatorCache,
                                      config::GRAPEConfig; ε::Float64=1e-6)
    n_steps = config.n_steps
    grad_J, grad_U, grad_Δ = zeros(n_steps), zeros(n_steps), zeros(n_steps)

    F0 = compute_fidelity(psi0, psi_target, J_vec, U_vec, Δ_vec, cache, config)

    for n in 1:n_steps-1
        J_plus = copy(J_vec); J_plus[n] += ε
        grad_J[n] = (compute_fidelity(psi0, psi_target, J_plus, U_vec, Δ_vec, cache, config) - F0) / ε

        U_plus = copy(U_vec); U_plus[n] += ε
        grad_U[n] = (compute_fidelity(psi0, psi_target, J_vec, U_plus, Δ_vec, cache, config) - F0) / ε

        Δ_plus = copy(Δ_vec); Δ_plus[n] += ε
        grad_Δ[n] = (compute_fidelity(psi0, psi_target, J_vec, U_vec, Δ_plus, cache, config) - F0) / ε
    end

    return grad_J, grad_U, grad_Δ, F0
end

# ============================================================================
# Optimization
# ============================================================================

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
            compute_gradients_analytical(psi0, psi_target, J, U, Δ, cache, config) :
            compute_gradients_numerical(psi0, psi_target, J, U, Δ, cache, config)

        if fidelity > best_fidelity
            best_fidelity = fidelity
            best_J, best_U, best_Δ = copy(J), copy(U), copy(Δ)
        end

        verbose && (iter == 1 || iter % 10 == 0) &&
            @printf("Iter %3d: fidelity = %.6f (best = %.6f)\n", iter, fidelity, best_fidelity)

        best_fidelity > 1.0 - config.tolerance && (verbose && println("Converged!"); break)

        if iter > 1
            lr = fidelity < prev_fidelity - 0.05 ? lr * 0.7 :
                 fidelity > prev_fidelity ? min(lr * 1.02, learning_rate) : lr
        end
        prev_fidelity = fidelity

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
            compute_gradients_analytical(psi0, psi_target, unpack(x)..., cache, config) :
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

    verbose && println("\nL-BFGS: $(Optim.iterations(result)) iterations")
    return J_opt, U_opt, Δ_opt, final_fidelity
end

# ============================================================================
# Analysis and testing
# ============================================================================

function analyze_noon_state(psi::MPS, s, config::GRAPEConfig)
    n = config.n_particles
    psi_left = MPS(s, vcat(["$n"], fill("0", config.n_sites - 1)))
    psi_right = MPS(s, vcat(fill("0", config.n_sites - 1), ["$n"]))
    target = make_noon_state(s, config)
    return (prob_left=abs2(inner(psi_left, psi)),
            prob_right=abs2(inner(psi_right, psi)),
            fidelity=abs2(inner(target, psi)))
end

function test_gradients(s, cache::OperatorCache, config::GRAPEConfig; verbose=true)
    Random.seed!(123)
    n_steps = config.n_steps
    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    J = 0.5 .+ 0.2 * randn(n_steps)
    U = 0.1 .+ 0.05 * randn(n_steps)
    Δ = 0.2 * randn(n_steps)

    grad_ana = compute_gradients_analytical(psi0, psi_target, J, U, Δ, cache, config)
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
    println("GRAPE Optimal Control (MPS) - Version 2")
    println("="^60)
    println("N=$N bosons, k=$k sites, T=$T, dt=$(@sprintf("%.3f", dt))")

    config = GRAPEConfig(k, N, num_steps, dt, 1e-8, 300, 1e-3)
    s = siteinds("Boson", k; dim=N+1, conserve_qns=true)

    println("\nBuilding operator cache...")
    cache = build_operator_cache(s, k)

    println("\nGradient verification:")
    test_gradients(s, cache, config)

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

    return J_opt, U_opt, Δ_opt, fidelity
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
