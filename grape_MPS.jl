"""
GRAPE (Gradient Ascent Pulse Engineering) for MPS quantum optimal control.

Optimizes control pulses (J, U, Δ) to maximize fidelity F = |⟨ψ_target|ψ(T)⟩|²
for a Bose-Hubbard system using Matrix Product States (MPS).

Default setup:
- Initial state: |N,0,...,0⟩ (all particles on first site)
- Target state: NOON state (|N,0,...,0⟩ + |0,...,0,N⟩)/√2

Hamiltonian:
- H1 (on-site): U·N(N-1) + Δ·(i-center)·N
- H2 (hopping): J·(A†_i·A_{i+1} + h.c.)

Uses 2nd-order Trotter decomposition with analytical gradients.
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
# Hamiltonian construction
# ============================================================================

"""Build on-site Hamiltonian: H1 = U·N(N-1) + Δ·(i-center)·N"""
function build_H1(site_idx::Int, U::Float64, Δ::Float64, s, n_sites::Int)
    center = (n_sites + 1) / 2
    h = U * op("N * N", s[site_idx]) - U * op("N", s[site_idx])
    h += Δ * (site_idx - center) * op("N", s[site_idx])
    return h
end

"""Build hopping Hamiltonian: H2 = J·(A†_i·A_{i+1} + h.c.)"""
function build_H2(site_idx::Int, J::Float64, s)
    return J * op("Adag", s[site_idx]) * op("A", s[site_idx + 1]) +
           J * op("A", s[site_idx]) * op("Adag", s[site_idx + 1])
end

"""∂H1/∂U = N(N-1)"""
function grad_H1_U(site_idx::Int, s)
    return op("N * N", s[site_idx]) - op("N", s[site_idx])
end

"""∂H1/∂Δ = (i - center)·N"""
function grad_H1_Δ(site_idx::Int, s, n_sites::Int)
    return (site_idx - (n_sites + 1) / 2) * op("N", s[site_idx])
end

"""∂H2/∂J = A†_i·A_{i+1} + A_i·A†_{i+1}"""
function grad_H2_J(site_idx::Int, s)
    return op("Adag", s[site_idx]) * op("A", s[site_idx + 1]) +
           op("A", s[site_idx]) * op("Adag", s[site_idx + 1])
end

# ============================================================================
# Gate construction
# ============================================================================

"""Build H1 gates for all sites."""
function make_H1_gates(U::Float64, Δ::Float64, dt_factor::Float64, s, n_sites::Int)
    return [exp(-im * dt_factor * build_H1(j, U, Δ, s, n_sites)) for j in 1:n_sites]
end

"""Build H2 gates for all bonds."""
function make_H2_gates(J::Float64, dt::Float64, s, n_sites::Int)
    return [exp(-im * dt * build_H2(j, J, s)) for j in 1:(n_sites-1)]
end

"""Build 2nd order Trotter gates: exp(-iH1·dt/2)·exp(-iH2·dt)·exp(-iH1·dt/2)"""
function make_trotter_gates(J::Float64, U::Float64, Δ::Float64, dt::Float64, s, n_sites::Int)
    gates = ITensor[]
    append!(gates, make_H1_gates(U, Δ, dt/2, s, n_sites))
    append!(gates, make_H2_gates(J, dt, s, n_sites))
    append!(gates, make_H1_gates(U, Δ, dt/2, s, n_sites))
    return gates
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
                           Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    psi = copy(psi0)
    for n in 1:config.n_steps-1
        gates = make_trotter_gates(J_vec[n], U_vec[n], Δ_vec[n], config.dt, s, config.n_sites)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
    end
    return psi
end

"""Compute fidelity F = |⟨ψ_target|ψ(T)⟩|²"""
function compute_fidelity(psi0::MPS, psi_target::MPS, J_vec::Vector{Float64},
                          U_vec::Vector{Float64}, Δ_vec::Vector{Float64},
                          s, config::GRAPEConfig)
    psi_final = forward_propagate(psi0, J_vec, U_vec, Δ_vec, s, config)
    return abs2(inner(psi_target, psi_final))
end

# ============================================================================
# Expectation values for gradient computation
# ============================================================================

"""Compute ⟨χ|H|ψ⟩ for single-site operator H at site j."""
function expect_single_site(chi::MPS, psi::MPS, H_op::ITensor, site_j::Int)
    psi_H = copy(psi)
    psi_H[site_j] = noprime(H_op * psi[site_j])
    return inner(chi, psi_H)
end

"""Compute ⟨χ|H|ψ⟩ for two-site operator H at sites j, j+1."""
function expect_two_site(chi::MPS, psi::MPS, H_op::ITensor, site_j::Int)
    n = length(psi)
    chi_orth = orthogonalize(chi, site_j)
    psi_orth = orthogonalize(psi, site_j)

    # Left environment
    L = site_j == 1 ? ITensor(1.0) :
        reduce((acc, i) -> acc * dag(chi_orth[i]) * psi_orth[i], 1:site_j-1; init=ITensor(1.0))

    # Right environment
    R = site_j + 1 == n ? ITensor(1.0) :
        reduce((acc, i) -> acc * dag(chi_orth[i]) * psi_orth[i], n:-1:site_j+2; init=ITensor(1.0))

    # Local contraction with operator
    psi_local = psi_orth[site_j] * psi_orth[site_j + 1]
    H_psi = noprime(H_op * psi_local)
    chi_local = dag(chi_orth[site_j]) * dag(chi_orth[site_j + 1])

    return scalar(L * chi_local * H_psi * R)
end

# ============================================================================
# Analytical gradient computation
# ============================================================================

"""
Forward propagation with gate-level state storage for gradient computation.
Returns: (psi_before, psi_after_H1, psi_after_H2) arrays.
"""
function forward_propagate_detailed(psi0::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                     Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    n_steps, n_sites, dt, cutoff = config.n_steps, config.n_sites, config.dt, config.cutoff

    psi_before = Vector{MPS}(undef, n_steps)
    psi_after_H1 = Vector{MPS}(undef, n_steps)
    psi_after_H2 = Vector{MPS}(undef, n_steps)

    psi = copy(psi0)
    for n in 1:n_steps-1
        psi_before[n] = copy(psi)

        # First H1 (dt/2)
        psi = apply(make_H1_gates(U_vec[n], Δ_vec[n], dt/2, s, n_sites), psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H1[n] = copy(psi)

        # H2 (dt)
        psi = apply(make_H2_gates(J_vec[n], dt, s, n_sites), psi; cutoff=cutoff)
        normalize!(psi)
        psi_after_H2[n] = copy(psi)

        # Second H1 (dt/2)
        psi = apply(make_H1_gates(U_vec[n], Δ_vec[n], dt/2, s, n_sites), psi; cutoff=cutoff)
        normalize!(psi)
    end

    psi_before[n_steps] = copy(psi)
    psi_after_H1[n_steps] = copy(psi)
    psi_after_H2[n_steps] = copy(psi)

    return psi_before, psi_after_H1, psi_after_H2
end

"""
Backward propagation with gate-level state storage.
Returns: (chi_after, chi_before_H2) arrays.
"""
function backward_propagate_detailed(chi_T::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                      Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    n_steps, n_sites, dt, cutoff = config.n_steps, config.n_sites, config.dt, config.cutoff

    chi_after = Vector{MPS}(undef, n_steps)
    chi_before_H2 = Vector{MPS}(undef, n_steps)

    chi = copy(chi_T)
    chi_after[n_steps] = copy(chi)
    chi_before_H2[n_steps] = copy(chi)

    for n in n_steps-1:-1:1
        # Backward through second H1 (adjoint)
        gates_H1_adj = [exp(+im * dt/2 * build_H1(j, U_vec[n], Δ_vec[n], s, n_sites))
                        for j in n_sites:-1:1]
        chi = apply(gates_H1_adj, chi; cutoff=cutoff)
        normalize!(chi)
        chi_before_H2[n+1] = copy(chi)

        # Backward through H2 (adjoint)
        gates_H2_adj = [exp(+im * dt * build_H2(j, J_vec[n], s)) for j in (n_sites-1):-1:1]
        chi = apply(gates_H2_adj, chi; cutoff=cutoff)
        normalize!(chi)

        # Backward through first H1 (adjoint)
        chi = apply(gates_H1_adj, chi; cutoff=cutoff)
        normalize!(chi)
        chi_after[n] = copy(chi)
    end
    chi_before_H2[1] = copy(chi)

    return chi_after, chi_before_H2
end

"""
Compute analytical GRAPE gradients.

Uses the formula: ∂F/∂θ = 2·Re[⟨target|ψ(T)⟩* · ⟨χ|(-i·dt·∂H/∂θ)|ψ⟩]
with proper Trotter decomposition handling.
"""
function compute_gradients_analytical(psi0::MPS, psi_target::MPS,
                                       J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                       Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    n_steps, n_sites, dt = config.n_steps, config.n_sites, config.dt

    grad_J = zeros(n_steps)
    grad_U = zeros(n_steps)
    grad_Δ = zeros(n_steps)

    # Forward and backward propagation
    psi_before, psi_after_H1, psi_after_H2 = forward_propagate_detailed(
        psi0, J_vec, U_vec, Δ_vec, s, config)
    chi_after, chi_before_H2 = backward_propagate_detailed(
        psi_target, J_vec, U_vec, Δ_vec, s, config)

    overlap = inner(psi_target, psi_before[n_steps])
    F = abs2(overlap)

    for n in 1:n_steps-1
        # J gradient (H2 gate)
        for j in 1:(n_sites-1)
            expect_val = expect_two_site(chi_before_H2[n+1], psi_after_H2[n], grad_H2_J(j, s), j)
            grad_J[n] += 2.0 * real(conj(overlap) * (-im * dt) * expect_val)
        end

        # U and Δ gradients (both H1 gates)
        chi_first, psi_first = chi_before_H2[n+1], psi_after_H1[n]
        chi_second, psi_second = chi_after[n+1], psi_before[n+1]

        for j in 1:n_sites
            # First H1 (dt/2)
            expect_U = expect_single_site(chi_first, psi_first, grad_H1_U(j, s), j)
            expect_Δ = expect_single_site(chi_first, psi_first, grad_H1_Δ(j, s, n_sites), j)
            grad_U[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_U)
            grad_Δ[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_Δ)

            # Second H1 (dt/2)
            expect_U = expect_single_site(chi_second, psi_second, grad_H1_U(j, s), j)
            expect_Δ = expect_single_site(chi_second, psi_second, grad_H1_Δ(j, s, n_sites), j)
            grad_U[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_U)
            grad_Δ[n] += 2.0 * real(conj(overlap) * (-im * dt/2) * expect_Δ)
        end
    end

    return grad_J, grad_U, grad_Δ, F
end

"""Compute numerical gradients using finite differences."""
function compute_gradients_numerical(psi0::MPS, psi_target::MPS,
                                      J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                      Δ_vec::Vector{Float64}, s, config::GRAPEConfig;
                                      ε::Float64=1e-6)
    n_steps = config.n_steps
    grad_J, grad_U, grad_Δ = zeros(n_steps), zeros(n_steps), zeros(n_steps)

    F0 = compute_fidelity(psi0, psi_target, J_vec, U_vec, Δ_vec, s, config)

    for n in 1:n_steps-1
        for (grad, vec, name) in [(grad_J, J_vec, :J), (grad_U, U_vec, :U), (grad_Δ, Δ_vec, :Δ)]
            vec_plus = copy(vec)
            vec_plus[n] += ε
            F_plus = if name == :J
                compute_fidelity(psi0, psi_target, vec_plus, U_vec, Δ_vec, s, config)
            elseif name == :U
                compute_fidelity(psi0, psi_target, J_vec, vec_plus, Δ_vec, s, config)
            else
                compute_fidelity(psi0, psi_target, J_vec, U_vec, vec_plus, s, config)
            end
            grad[n] = (F_plus - F0) / ε
        end
    end

    return grad_J, grad_U, grad_Δ, F0
end

# ============================================================================
# Optimization
# ============================================================================

"""
GRAPE optimization using gradient descent with momentum.

Returns: (J_opt, U_opt, Δ_opt, final_fidelity)
"""
function grape_optimize(s, config::GRAPEConfig;
                        J0=nothing, U0=nothing, Δ0=nothing,
                        learning_rate=0.1, momentum=0.9,
                        use_analytical=true, verbose=true)
    n_steps = config.n_steps

    # Initialize controls
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
            compute_gradients_analytical(psi0, psi_target, J, U, Δ, s, config) :
            compute_gradients_numerical(psi0, psi_target, J, U, Δ, s, config)

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
function grape_optimize_lbfgs(s, config::GRAPEConfig;
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

    objective(x) = 1.0 - compute_fidelity(psi0, psi_target, unpack(x)..., s, config)

    function gradient!(g, x)
        grad_J, grad_U, grad_Δ, _ = use_analytical ?
            compute_gradients_analytical(psi0, psi_target, unpack(x)..., s, config) :
            compute_gradients_numerical(psi0, psi_target, unpack(x)..., s, config)
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
function test_gradients(s, config::GRAPEConfig; verbose=true)
    Random.seed!(123)
    n_steps = config.n_steps

    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    J = 0.5 .+ 0.2 * randn(n_steps)
    U = 0.1 .+ 0.05 * randn(n_steps)
    Δ = 0.2 * randn(n_steps)

    grad_ana = compute_gradients_analytical(psi0, psi_target, J, U, Δ, s, config)
    grad_num = compute_gradients_numerical(psi0, psi_target, J, U, Δ, s, config)

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
    num_steps = 51
    dt = T / (num_steps - 1)

    println("="^60)
    println("GRAPE Optimal Control (MPS) for Bose-Hubbard Model")
    println("="^60)
    println("N=$N bosons, k=$k sites, T=$T, dt=$(@sprintf("%.3f", dt))")

    config = GRAPEConfig(k, N, num_steps, dt, 1e-8, 300, 1e-3)
    s = siteinds("Boson", k; dim=N+1, conserve_qns=true)

    # Test gradients
    println("\nGradient verification:")
    test_gradients(s, config)

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

    J_opt, U_opt, Δ_opt, fidelity = grape_optimize(s, config; J0=J0, U0=U0, Δ0=Δ0)

    println("\n" * "-"^60)
    @printf("Final fidelity: %.6f\n", fidelity)

    psi_final = forward_propagate(psi0, J_opt, U_opt, Δ_opt, s, config)
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
