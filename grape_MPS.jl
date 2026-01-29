"""
GRAPE (Gradient Ascent Pulse Engineering) for MPS quantum optimal control.

Goal: Maximize fidelity F = |⟨ψ_target|ψ(T)⟩|²
Starting from: |N,0,0⟩ (all particles on first site)
Target: NOON state (|N,0,0⟩ + |0,0,N⟩)/√2

Gradient formula:
∂F/∂θ_n = 2·Re[⟨ψ_target|ψ(T)⟩* · ⟨χ_{n+1}|(-i·dt·∂H/∂θ)|ψ_n⟩]

where:
- ψ_n = forward state after n-1 gates (before gate n)
- χ_{n+1} = backward costate = U†_{n+1}...U†_T |ψ_target⟩
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Random
using Printf

# ============================================================================
# Configuration
# ============================================================================

struct GRAPEConfig
    n_sites::Int
    n_particles::Int
    n_steps::Int          # Number of time steps
    dt::Float64           # Time step size
    cutoff::Float64       # MPS truncation cutoff
    learning_rate::Float64
    max_iterations::Int
    convergence_threshold::Float64
end

# ============================================================================
# Hamiltonian building functions
# ============================================================================

"""Build on-site Hamiltonian H1 = U·N(N-1) + Δ·(i-center)·N"""
function build_H1(site_idx::Int, U::Float64, Δ::Float64, s, n_sites::Int)
    center = (n_sites + 1) / 2
    h = U * op("N * N", s[site_idx]) + U * (-1.0) * op("N", s[site_idx])
    h += Δ * (site_idx - center) * op("N", s[site_idx])
    return h
end

"""Build hopping Hamiltonian H2 = J·(A†_i·A_{i+1} + h.c.)"""
function build_H2(site_idx::Int, J::Float64, s)
    h = J * op("Adag", s[site_idx]) * op("A", s[site_idx + 1])
    h += J * op("A", s[site_idx]) * op("Adag", s[site_idx + 1])
    return h
end

# ============================================================================
# Hamiltonian gradients (∂H/∂θ)
# ============================================================================

"""∂H1/∂U = N(N-1)"""
function grad_H1_U(site_idx::Int, s)
    return op("N * N", s[site_idx]) + (-1.0) * op("N", s[site_idx])
end

"""∂H1/∂Δ = (i - center)·N"""
function grad_H1_Δ(site_idx::Int, s, n_sites::Int)
    center = (n_sites + 1) / 2
    return (site_idx - center) * op("N", s[site_idx])
end

"""∂H2/∂J = A†_i·A_{i+1} + A_i·A†_{i+1}"""
function grad_H2_J(site_idx::Int, s)
    h = op("Adag", s[site_idx]) * op("A", s[site_idx + 1])
    h += op("A", s[site_idx]) * op("Adag", s[site_idx + 1])
    return h
end

# ============================================================================
# Gate construction
# ============================================================================

"""Build 2nd order Trotter gates for one time step."""
function make_trotter_gates(J::Float64, U::Float64, Δ::Float64, dt::Float64,
                            s, n_sites::Int)
    gates = ITensor[]

    # First half: exp(-i·H1·dt/2) for all sites
    for j in 1:n_sites
        hj = build_H1(j, U, Δ, s, n_sites)
        push!(gates, exp(-im * dt/2 * hj))
    end

    # Middle: exp(-i·H2·dt) for all bonds
    for j in 1:(n_sites - 1)
        hj = build_H2(j, J, s)
        push!(gates, exp(-im * dt * hj))
    end

    # Second half: exp(-i·H1·dt/2) for all sites
    for j in 1:n_sites
        hj = build_H1(j, U, Δ, s, n_sites)
        push!(gates, exp(-im * dt/2 * hj))
    end

    return gates
end

"""Build adjoint gates for backward propagation (reverse order)."""
function make_trotter_gates_adjoint(J::Float64, U::Float64, Δ::Float64, dt::Float64,
                                    s, n_sites::Int)
    gates = ITensor[]

    # Reverse of forward: second half first
    for j in n_sites:-1:1
        hj = build_H1(j, U, Δ, s, n_sites)
        push!(gates, exp(+im * dt/2 * hj))
    end

    # Middle (reversed)
    for j in (n_sites - 1):-1:1
        hj = build_H2(j, J, s)
        push!(gates, exp(+im * dt * hj))
    end

    # First half (reversed)
    for j in n_sites:-1:1
        hj = build_H1(j, U, Δ, s, n_sites)
        push!(gates, exp(+im * dt/2 * hj))
    end

    return gates
end

# ============================================================================
# State initialization
# ============================================================================

"""Create initial state: all particles on first site |N,0,...,0⟩"""
function make_initial_state(s, config::GRAPEConfig)
    state = ["$(config.n_particles)"]
    for _ in 2:config.n_sites
        push!(state, "0")
    end
    return MPS(s, state)
end

"""Create NOON target state: (|N,0,...,0⟩ + |0,...,0,N⟩)/√2"""
function make_noon_state(s, config::GRAPEConfig)
    n = config.n_particles

    # |N,0,...,0⟩
    state1 = ["$n"]
    for _ in 2:config.n_sites
        push!(state1, "0")
    end
    psi1 = MPS(s, state1)

    # |0,...,0,N⟩
    state2 = String[]
    for _ in 1:(config.n_sites - 1)
        push!(state2, "0")
    end
    push!(state2, "$n")
    psi2 = MPS(s, state2)

    # Superposition
    noon = add(psi1, psi2; cutoff=config.cutoff)
    noon ./= sqrt(2)
    normalize!(noon)

    return noon
end

# ============================================================================
# GRAPE core functions
# ============================================================================

"""
Forward propagation: evolve initial state through all time steps.
Returns array of states [ψ_0, ψ_1, ..., ψ_T] where ψ_n is state BEFORE step n+1.
"""
function forward_propagate(J_vec::Vector{Float64}, U_vec::Vector{Float64},
                           Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    psi = make_initial_state(s, config)
    psi_history = MPS[copy(psi)]  # ψ_0 = initial state

    for n in 1:config.n_steps
        gates = make_trotter_gates(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                   s, config.n_sites)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
        push!(psi_history, copy(psi))  # ψ_n = state after step n
    end

    return psi_history
end

"""
Backward propagation: evolve target state backward through all time steps.
Returns array [χ_T, χ_{T-1}, ..., χ_1, χ_0] where χ_n = U†_n...U†_T |target⟩
Note: χ_{n+1} in gradient formula corresponds to chi_history[T-n] due to indexing.
"""
function backward_propagate(J_vec::Vector{Float64}, U_vec::Vector{Float64},
                            Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    chi = make_noon_state(s, config)
    chi_history = MPS[copy(chi)]  # χ_T = target

    # Propagate backward: apply U†_T, U†_{T-1}, ..., U†_1
    for n in config.n_steps:-1:1
        gates = make_trotter_gates_adjoint(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                           s, config.n_sites)
        chi = apply(gates, chi; cutoff=config.cutoff)
        normalize!(chi)
        push!(chi_history, copy(chi))
    end

    return chi_history  # [χ_T, χ_{T-1}, ..., χ_0]
end

"""
Compute ⟨χ|O|ψ⟩ for one-site operator O at site_idx.
"""
function expectation_one_site(chi::MPS, psi::MPS, O::ITensor, site_idx::Int)
    psi_new = copy(psi)
    psi_new[site_idx] = noprime(O * psi[site_idx])
    return inner(chi, psi_new)
end

"""
Compute ⟨χ|O|ψ⟩ for two-site operator O at sites (site_idx, site_idx+1).
"""
function expectation_two_site(chi::MPS, psi::MPS, O::ITensor, site_idx::Int)
    n = length(psi)

    # Contract from left
    if site_idx > 1
        result = dag(chi[1]) * psi[1]
        for j in 2:(site_idx - 1)
            result = result * dag(chi[j]) * psi[j]
        end
    else
        result = ITensor(1.0)
    end

    # Two-site block with operator
    psi_two = psi[site_idx] * psi[site_idx + 1]
    O_psi = O * psi_two
    chi_i = prime(chi[site_idx], "Site")
    chi_i1 = prime(chi[site_idx + 1], "Site")
    chi_two = dag(chi_i) * dag(chi_i1)
    result = result * chi_two * O_psi

    # Contract from right
    if site_idx + 1 < n
        for j in (site_idx + 2):n
            result = result * dag(chi[j]) * psi[j]
        end
    end

    return scalar(result)
end

"""
Compute GRAPE gradients for all parameters at all time steps.

Gradient formula:
∂F/∂θ_n = 2·Re[⟨target|ψ_T⟩* · ⟨χ_{n+1}|(-i·dt·∂H/∂θ)|ψ_n⟩]

where ψ_n is state before step n, χ_{n+1} = U†_{n+1}...U†_T|target⟩
"""
function compute_grape_gradients(J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                  Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    # Forward propagation
    psi_history = forward_propagate(J_vec, U_vec, Δ_vec, s, config)

    # Backward propagation
    chi_history = backward_propagate(J_vec, U_vec, Δ_vec, s, config)

    # Final overlap ⟨target|ψ_T⟩
    target = make_noon_state(s, config)
    psi_T = psi_history[end]
    overlap = inner(target, psi_T)
    fidelity = abs2(overlap)

    # Initialize gradient arrays
    dF_dJ = zeros(Float64, config.n_steps)
    dF_dU = zeros(Float64, config.n_steps)
    dF_dΔ = zeros(Float64, config.n_steps)

    dt = config.dt

    # Compute gradients for each time step
    for n in 1:config.n_steps
        # ψ_n = state before step n (index n in psi_history since Julia is 1-indexed)
        # Actually psi_history[1] = ψ_0, psi_history[n+1] = ψ_n (after n steps)
        # So ψ_{n-1} (before step n) is psi_history[n]
        psi_n = psi_history[n]

        # χ_{n+1} = state after backward propagating through steps n+1,...,T
        # chi_history[1] = χ_T = target
        # chi_history[2] = χ_{T-1} = U†_T |target⟩
        # chi_history[k+1] = χ_{T-k}
        # We need χ_{n+1}, which means k = T - (n+1) = T - n - 1
        # So index = k + 1 = T - n
        # But wait, let me recount...
        # chi_history[1] = target (before any backward steps)
        # After backward step T: chi_history[2] = U†_T |target⟩ = χ_{T-1}
        # After backward step T-1: chi_history[3] = U†_{T-1} U†_T |target⟩ = χ_{T-2}
        # ...
        # After backward step n+1: chi_history[T-n+1] = U†_{n+1}...U†_T |target⟩ = χ_n
        # So χ_{n} corresponds to chi_history[T - n + 1]
        # We need χ_{n+1} which is chi_history[T - (n+1) + 1] = chi_history[T - n]
        # But actually I think my indexing in backward_propagate is:
        # chi_history[1] = initial (target)
        # chi_history[i+1] = after applying gates for step T-i+1
        # Let me think again...

        # Actually, looking at backward_propagate:
        # chi_history = [target]  (index 1)
        # loop n from T to 1:
        #   apply U†_n
        #   push chi (now index 2 after first iteration, etc.)
        # So chi_history[1] = target = |ψ_target⟩
        # chi_history[2] = U†_T |target⟩
        # chi_history[3] = U†_{T-1} U†_T |target⟩
        # chi_history[T-n+2] = U†_{n}...U†_T |target⟩

        # For gradient at step n, we need:
        # χ_{n} = U†_{n+1}...U†_T |target⟩ (costate AFTER step n, or equivalently BEFORE step n in backward direction)
        # This is chi_history[T - n + 1]

        chi_n = chi_history[config.n_steps - n + 1]

        # Compute gradient contributions
        grad_J_n = 0.0 + 0.0im
        grad_U_n = 0.0 + 0.0im
        grad_Δ_n = 0.0 + 0.0im

        # One-site terms (U and Δ) - factor of 2 for two half-steps
        for j in 1:config.n_sites
            dH_dU = grad_H1_U(j, s)
            dH_dΔ = grad_H1_Δ(j, s, config.n_sites)

            grad_U_n += -im * dt * expectation_one_site(chi_n, psi_n, dH_dU, j)
            grad_Δ_n += -im * dt * expectation_one_site(chi_n, psi_n, dH_dΔ, j)
        end

        # Two-site terms (J)
        for j in 1:(config.n_sites - 1)
            dH_dJ = grad_H2_J(j, s)
            grad_J_n += -im * dt * expectation_two_site(chi_n, psi_n, dH_dJ, j)
        end

        # Apply the fidelity gradient formula: ∂F/∂θ = 2·Re[⟨target|ψ_T⟩* · ...]
        dF_dJ[n] = 2 * real(conj(overlap) * grad_J_n)
        dF_dU[n] = 2 * real(conj(overlap) * grad_U_n)
        dF_dΔ[n] = 2 * real(conj(overlap) * grad_Δ_n)
    end

    return (dF_dJ=dF_dJ, dF_dU=dF_dU, dF_dΔ=dF_dΔ, fidelity=fidelity, overlap=overlap)
end

"""
Compute fidelity only (for evaluation).
"""
function compute_fidelity(J_vec::Vector{Float64}, U_vec::Vector{Float64},
                          Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    psi = make_initial_state(s, config)

    for n in 1:config.n_steps
        gates = make_trotter_gates(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                   s, config.n_sites)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
    end

    target = make_noon_state(s, config)
    return abs2(inner(target, psi))
end

# ============================================================================
# GRAPE optimization
# ============================================================================

"""
Compute numerical gradients using finite differences (more accurate but slower).
"""
function compute_numerical_gradients(J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                      Δ_vec::Vector{Float64}, s, config::GRAPEConfig;
                                      ε::Float64=1e-6)
    n_steps = config.n_steps
    dF_dJ = zeros(Float64, n_steps)
    dF_dU = zeros(Float64, n_steps)
    dF_dΔ = zeros(Float64, n_steps)

    for n in 1:n_steps
        # J gradient
        J_plus = copy(J_vec); J_plus[n] += ε
        J_minus = copy(J_vec); J_minus[n] -= ε
        dF_dJ[n] = (compute_fidelity(J_plus, U_vec, Δ_vec, s, config) -
                    compute_fidelity(J_minus, U_vec, Δ_vec, s, config)) / (2ε)

        # U gradient
        U_plus = copy(U_vec); U_plus[n] += ε
        U_minus = copy(U_vec); U_minus[n] -= ε
        dF_dU[n] = (compute_fidelity(J_vec, U_plus, Δ_vec, s, config) -
                    compute_fidelity(J_vec, U_minus, Δ_vec, s, config)) / (2ε)

        # Δ gradient
        Δ_plus = copy(Δ_vec); Δ_plus[n] += ε
        Δ_minus = copy(Δ_vec); Δ_minus[n] -= ε
        dF_dΔ[n] = (compute_fidelity(J_vec, U_vec, Δ_plus, s, config) -
                    compute_fidelity(J_vec, U_vec, Δ_minus, s, config)) / (2ε)
    end

    fidelity = compute_fidelity(J_vec, U_vec, Δ_vec, s, config)
    return (dF_dJ=dF_dJ, dF_dU=dF_dU, dF_dΔ=dF_dΔ, fidelity=fidelity)
end

"""
Run GRAPE optimization to maximize fidelity.
use_numerical_gradients: if true, use finite differences (slower but more accurate)
"""
function run_grape(config::GRAPEConfig; verbose::Bool=true, use_numerical_gradients::Bool=false)
    # Create site indices
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

    # Initialize control parameters
    J_vec = 0.5 * ones(config.n_steps)   # Uniform hopping
    U_vec = zeros(config.n_steps)         # No interaction initially
    Δ_vec = zeros(config.n_steps)         # No tilt initially

    # History for plotting
    fidelity_history = Float64[]

    if verbose
        println("="^60)
        println("GRAPE Optimization")
        println("="^60)
        println("Sites: $(config.n_sites), Particles: $(config.n_particles)")
        println("Time steps: $(config.n_steps), dt: $(config.dt)")
        println("Total time T = $(config.n_steps * config.dt)")
        println("Learning rate: $(config.learning_rate)")
        println("Gradient method: $(use_numerical_gradients ? "Numerical" : "Adjoint")")
        println("="^60)
    end

    best_fidelity = 0.0
    best_J = copy(J_vec)
    best_U = copy(U_vec)
    best_Δ = copy(Δ_vec)

    learning_rate = config.learning_rate
    prev_fidelity = 0.0

    for iter in 1:config.max_iterations
        # Compute gradients
        if use_numerical_gradients
            result = compute_numerical_gradients(J_vec, U_vec, Δ_vec, s, config)
        else
            result = compute_grape_gradients(J_vec, U_vec, Δ_vec, s, config)
        end
        fidelity = result.fidelity
        push!(fidelity_history, fidelity)

        # Track best
        if fidelity > best_fidelity
            best_fidelity = fidelity
            best_J = copy(J_vec)
            best_U = copy(U_vec)
            best_Δ = copy(Δ_vec)
        end

        # Print progress
        if verbose && (iter == 1 || iter % 10 == 0 || iter == config.max_iterations)
            @printf("Iter %4d: Fidelity = %.6f (lr = %.4f)\n", iter, fidelity, learning_rate)
        end

        # Check convergence
        if fidelity > 1.0 - config.convergence_threshold
            if verbose
                println("\nConverged! Fidelity reached $(fidelity)")
            end
            break
        end

        # Adaptive learning rate
        if iter > 1 && fidelity < prev_fidelity - 0.01
            learning_rate *= 0.5  # Reduce if fidelity dropped significantly
        elseif iter > 1 && fidelity > prev_fidelity
            learning_rate = min(learning_rate * 1.1, config.learning_rate)  # Increase if improving
        end
        prev_fidelity = fidelity

        # Gradient ascent update (maximize fidelity)
        J_vec .+= learning_rate .* result.dF_dJ
        U_vec .+= learning_rate .* result.dF_dU
        Δ_vec .+= learning_rate .* result.dF_dΔ

        # Keep J positive
        J_vec = max.(J_vec, 0.01)
    end

    if verbose
        println("\n" * "="^60)
        println("Optimization complete!")
        println("Best fidelity: $(best_fidelity)")
        println("="^60)
    end

    return (J=best_J, U=best_U, Δ=best_Δ,
            fidelity=best_fidelity,
            history=fidelity_history,
            sites=s, config=config)
end

"""
Validate GRAPE gradients against finite differences.
"""
function validate_grape_gradients(config::GRAPEConfig; ε::Float64=1e-6, step_idx::Int=1)
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

    # Random control parameters
    Random.seed!(42)
    J_vec = 0.5 .+ 0.1 * randn(config.n_steps)
    U_vec = 0.1 * randn(config.n_steps)
    Δ_vec = 0.1 * randn(config.n_steps)

    println("=== Validating GRAPE Gradients ===")
    println("Step index: $step_idx, ε = $ε\n")

    # Analytical gradients
    result = compute_grape_gradients(J_vec, U_vec, Δ_vec, s, config)

    # Numerical gradients via finite differences
    # J gradient
    J_plus = copy(J_vec); J_plus[step_idx] += ε
    J_minus = copy(J_vec); J_minus[step_idx] -= ε
    F_J_plus = compute_fidelity(J_plus, U_vec, Δ_vec, s, config)
    F_J_minus = compute_fidelity(J_minus, U_vec, Δ_vec, s, config)
    num_dF_dJ = (F_J_plus - F_J_minus) / (2ε)

    # U gradient
    U_plus = copy(U_vec); U_plus[step_idx] += ε
    U_minus = copy(U_vec); U_minus[step_idx] -= ε
    F_U_plus = compute_fidelity(J_vec, U_plus, Δ_vec, s, config)
    F_U_minus = compute_fidelity(J_vec, U_minus, Δ_vec, s, config)
    num_dF_dU = (F_U_plus - F_U_minus) / (2ε)

    # Δ gradient
    Δ_plus = copy(Δ_vec); Δ_plus[step_idx] += ε
    Δ_minus = copy(Δ_vec); Δ_minus[step_idx] -= ε
    F_Δ_plus = compute_fidelity(J_vec, U_vec, Δ_plus, s, config)
    F_Δ_minus = compute_fidelity(J_vec, U_vec, Δ_minus, s, config)
    num_dF_dΔ = (F_Δ_plus - F_Δ_minus) / (2ε)

    println("∂F/∂J[$step_idx]:")
    println("  GRAPE:     $(result.dF_dJ[step_idx])")
    println("  Numerical: $num_dF_dJ")
    println("  Diff:      $(abs(result.dF_dJ[step_idx] - num_dF_dJ))")

    println("\n∂F/∂U[$step_idx]:")
    println("  GRAPE:     $(result.dF_dU[step_idx])")
    println("  Numerical: $num_dF_dU")
    println("  Diff:      $(abs(result.dF_dU[step_idx] - num_dF_dU))")

    println("\n∂F/∂Δ[$step_idx]:")
    println("  GRAPE:     $(result.dF_dΔ[step_idx])")
    println("  Numerical: $num_dF_dΔ")
    println("  Diff:      $(abs(result.dF_dΔ[step_idx] - num_dF_dΔ))")

    return (grape=(dF_dJ=result.dF_dJ[step_idx], dF_dU=result.dF_dU[step_idx], dF_dΔ=result.dF_dΔ[step_idx]),
            numerical=(dF_dJ=num_dF_dJ, dF_dU=num_dF_dU, dF_dΔ=num_dF_dΔ))
end

# ============================================================================
# Test optimal pulses from BM_GRAPE_trotter.jl using MPS
# ============================================================================

"""
Load optimal control pulses from text files generated by BM_GRAPE_trotter.jl.
"""
function load_optimal_pulses(base_path::String=".")
    J_opt = Float64[]
    U_opt = Float64[]
    Δ_opt = Float64[]

    open(joinpath(base_path, "J_opt_trotter.txt"), "r") do f
        for line in eachline(f)
            push!(J_opt, parse(Float64, strip(line)))
        end
    end

    open(joinpath(base_path, "U_opt_trotter.txt"), "r") do f
        for line in eachline(f)
            push!(U_opt, parse(Float64, strip(line)))
        end
    end

    open(joinpath(base_path, "Delta_opt_trotter.txt"), "r") do f
        for line in eachline(f)
            push!(Δ_opt, parse(Float64, strip(line)))
        end
    end

    return J_opt, U_opt, Δ_opt
end

"""
Forward propagation using optimal pulses (simpler version without storing history).
Returns the final MPS state.
"""
function forward_propagate_final(J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                  Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    psi = make_initial_state(s, config)

    for n in 1:config.n_steps
        gates = make_trotter_gates(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                   s, config.n_sites)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
    end

    return psi
end

"""
Analyze MPS state to check if it's a NOON state.
Returns occupation probabilities and other diagnostics.
"""
function analyze_noon_state(psi::MPS, s, config::GRAPEConfig)
    # Get particle number expectations at each site
    n_expect = expect(psi, "N")

    # Get particle number variance at each site
    n2_expect = expect(psi, "N * N")
    n_var = n2_expect .- n_expect.^2

    # Create basis states for overlap calculation
    n = config.n_particles

    # |N,0,...,0⟩
    state1 = ["$n"]
    for _ in 2:config.n_sites
        push!(state1, "0")
    end
    psi_left = MPS(s, state1)

    # |0,...,0,N⟩
    state2 = String[]
    for _ in 1:(config.n_sites - 1)
        push!(state2, "0")
    end
    push!(state2, "$n")
    psi_right = MPS(s, state2)

    # Compute overlaps
    overlap_left = inner(psi_left, psi)
    overlap_right = inner(psi_right, psi)

    # Probabilities
    prob_left = abs2(overlap_left)
    prob_right = abs2(overlap_right)

    # NOON state fidelity
    target = make_noon_state(s, config)
    fidelity = abs2(inner(target, psi))

    return (
        n_expect = n_expect,
        n_var = n_var,
        overlap_left = overlap_left,
        overlap_right = overlap_right,
        prob_left = prob_left,
        prob_right = prob_right,
        fidelity = fidelity
    )
end

# ============================================================================
# Main execution: Test optimal pulses from BM_GRAPE_trotter.jl
# ============================================================================

println("="^60)
println("Testing Optimal Pulses from BM_GRAPE_trotter.jl using MPS")
println("="^60)

# Load optimal pulses (1001 values, use first 1000 for 1000 Trotter steps)
println("\nLoading optimal control pulses...")
J_opt, U_opt, Δ_opt = load_optimal_pulses()
println("Loaded $(length(J_opt)) pulse values")

# Configuration matching BM_GRAPE_trotter.jl:
# T = 10, num_steps = 1001, dt = T/(num_steps-1) = 0.01
# But we use n_steps = 1000 (the number of actual Trotter steps)
config = GRAPEConfig(
    3,      # n_sites (matching BM_GRAPE_trotter.jl)
    3,      # n_particles (matching BM_GRAPE_trotter.jl)
    1000,   # n_steps (1000 Trotter steps, uses indices 1:1000)
    0.01,   # dt (total time = 10.0)
    1e-8,   # cutoff
    2.0,    # learning_rate (not used for testing)
    100,    # max_iterations (not used for testing)
    1e-6    # convergence_threshold (not used for testing)
)

# Create site indices
s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

println("\nSystem configuration:")
println("  Sites: $(config.n_sites)")
println("  Particles: $(config.n_particles)")
println("  Time steps: $(config.n_steps)")
println("  dt: $(config.dt)")
println("  Total time T = $(config.n_steps * config.dt)")

# Check initial state overlap with NOON
println("\n" * "-"^60)
println("Initial State Analysis")
println("-"^60)
initial_state = make_initial_state(s, config)
target = make_noon_state(s, config)
initial_fidelity = abs2(inner(target, initial_state))
println("Initial state |$(config.n_particles),0,0⟩")
println("Target: NOON state (|$(config.n_particles),0,0⟩ + |0,0,$(config.n_particles)⟩)/√2")
println("Initial fidelity with NOON: $initial_fidelity")
println("(Expected: 0.5 since |⟨NOON|N,0,0⟩|² = |1/√2|² = 0.5)")

# Forward propagate using optimal pulses
println("\n" * "-"^60)
println("MPS Forward Propagation with Optimal Pulses")
println("-"^60)
println("Propagating initial state using optimal pulses from BM_GRAPE_trotter.jl...")

# Use first 1000 pulse values (indices 1:1000 for 1000 Trotter steps)
J_test = J_opt[1:config.n_steps]
U_test = U_opt[1:config.n_steps]
Δ_test = Δ_opt[1:config.n_steps]

psi_final = forward_propagate_final(J_test, U_test, Δ_test, s, config)

# Analyze final state
println("\n" * "-"^60)
println("Final State Analysis")
println("-"^60)

result = analyze_noon_state(psi_final, s, config)

println("\nParticle number expectation ⟨N_i⟩ at each site:")
for (i, n) in enumerate(result.n_expect)
    @printf("  Site %d: ⟨N⟩ = %.6f\n", i, real(n))
end

println("\nParticle number variance Var(N_i) at each site:")
for (i, v) in enumerate(result.n_var)
    @printf("  Site %d: Var(N) = %.6f\n", i, real(v))
end

println("\nOverlaps with basis states:")
@printf("  ⟨%d,0,0|ψ_final⟩ = %.6f + %.6fi\n",
        config.n_particles, real(result.overlap_left), imag(result.overlap_left))
@printf("  ⟨0,0,%d|ψ_final⟩ = %.6f + %.6fi\n",
        config.n_particles, real(result.overlap_right), imag(result.overlap_right))

println("\nProbabilities:")
@printf("  |⟨%d,0,0|ψ_final⟩|² = %.6f\n", config.n_particles, result.prob_left)
@printf("  |⟨0,0,%d|ψ_final⟩|² = %.6f\n", config.n_particles, result.prob_right)
@printf("  Sum = %.6f (should be ≈1 for NOON state)\n", result.prob_left + result.prob_right)

println("\n" * "="^60)
println("NOON STATE FIDELITY")
println("="^60)
@printf("\n  F = |⟨NOON|ψ_final⟩|² = %.8f\n", result.fidelity)

if result.fidelity > 0.99
    println("\n  ✓ SUCCESS: Final MPS is a NOON state with high fidelity!")
elseif result.fidelity > 0.9
    println("\n  ~ GOOD: Final MPS approximates NOON state well.")
else
    println("\n  ✗ Final state does not match NOON state.")
end

println("\n" * "="^60)
println("COMPARISON: MPS vs State Vector Results")
println("="^60)
println("""
BM_GRAPE_trotter.jl (state vector) results:
  |⟨ψ_target|ψ_final⟩|² = 0.9999999963909736
  |ψ_final[1]|² = 0.5000131092900976  (|3,0,0⟩ component)
  |ψ_final[end]|² = 0.4999868872813438  (|0,0,3⟩ component)

MPS results:
  Fidelity = $(result.fidelity)
  |⟨3,0,0|ψ⟩|² = $(result.prob_left)
  |⟨0,0,3|ψ⟩|² = $(result.prob_right)
""")
println("="^60)
