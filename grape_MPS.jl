"""
GRAPE (Gradient Ascent Pulse Engineering) for MPS quantum optimal control.

Goal: Maximize fidelity F = |⟨ψ_target|ψ(T)⟩|²
Starting from: |N,0,0⟩ (all particles on first site)
Target: NOON state (|N,0,0⟩ + |0,0,N⟩)/√2

This implements the same GRAPE algorithm as BM_GRAPE_trotter.jl but uses
MPS (Matrix Product States) instead of state vectors for scalability.

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
using Optim

# ============================================================================
# Configuration
# ============================================================================

struct GRAPEConfig
    n_sites::Int
    n_particles::Int
    n_steps::Int          # Number of time steps
    dt::Float64           # Time step size
    cutoff::Float64       # MPS truncation cutoff
    max_iterations::Int
    tolerance::Float64    # Convergence tolerance (infidelity threshold)
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
# MPS Propagation functions
# ============================================================================

"""
Forward propagation: evolve initial state through all time steps.
Returns array of states [ψ_0, ψ_1, ..., ψ_{n_steps}] where ψ_n is state AFTER step n.
"""
function forward_propagate_store(psi0::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                  Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    n_steps = length(J_vec)
    psi_states = Vector{MPS}(undef, n_steps)

    psi = copy(psi0)
    psi_states[1] = copy(psi)

    for n in 1:n_steps-1
        gates = make_trotter_gates(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                   s, config.n_sites)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
        psi_states[n+1] = copy(psi)
    end

    return psi_states
end

"""
Forward propagation without storing intermediate states.
Returns only the final state.
"""
function forward_propagate(psi0::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                           Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    n_steps = length(J_vec)
    psi = copy(psi0)

    for n in 1:n_steps-1
        gates = make_trotter_gates(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                   s, config.n_sites)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
    end

    return psi
end

"""
Backward propagation: evolve target state backward through all time steps.
Returns array of costates.
"""
function backward_propagate_store(chi_T::MPS, J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                   Δ_vec::Vector{Float64}, s, config::GRAPEConfig)
    n_steps = length(J_vec)
    chi_states = Vector{MPS}(undef, n_steps)

    chi = copy(chi_T)
    chi_states[n_steps] = copy(chi)

    # Backward propagation uses adjoint of forward propagator
    for n in n_steps-1:-1:1
        gates = make_trotter_gates_adjoint(J_vec[n], U_vec[n], Δ_vec[n], config.dt,
                                           s, config.n_sites)
        chi = apply(gates, chi; cutoff=config.cutoff)
        normalize!(chi)
        chi_states[n] = copy(chi)
    end

    return chi_states
end

# ============================================================================
# Gradient computation using numerical differentiation
# ============================================================================

"""
Compute fidelity for given control parameters.
"""
function compute_fidelity_mps(psi0::MPS, psi_target::MPS, J_vec::Vector{Float64},
                               U_vec::Vector{Float64}, Δ_vec::Vector{Float64},
                               s, config::GRAPEConfig)
    psi_final = forward_propagate(psi0, J_vec, U_vec, Δ_vec, s, config)
    return abs2(inner(psi_target, psi_final))
end

"""
Compute GRAPE gradients using numerical differentiation (finite differences).
This is more robust for MPS with quantum number conservation.
"""
function compute_mps_gradients_numerical(psi0::MPS, psi_target::MPS,
                                          J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                          Δ_vec::Vector{Float64}, s, config::GRAPEConfig;
                                          ε::Float64=1e-5)
    n_steps = length(J_vec)

    grad_J = zeros(n_steps)
    grad_U = zeros(n_steps)
    grad_Delta = zeros(n_steps)

    # Base fidelity
    F0 = compute_fidelity_mps(psi0, psi_target, J_vec, U_vec, Δ_vec, s, config)

    # Compute gradients using forward differences (faster than central differences)
    for n in 1:n_steps-1
        # J gradient
        J_plus = copy(J_vec)
        J_plus[n] += ε
        F_J_plus = compute_fidelity_mps(psi0, psi_target, J_plus, U_vec, Δ_vec, s, config)
        grad_J[n] = (F_J_plus - F0) / ε

        # U gradient
        U_plus = copy(U_vec)
        U_plus[n] += ε
        F_U_plus = compute_fidelity_mps(psi0, psi_target, J_vec, U_plus, Δ_vec, s, config)
        grad_U[n] = (F_U_plus - F0) / ε

        # Delta gradient
        Delta_plus = copy(Δ_vec)
        Delta_plus[n] += ε
        F_Delta_plus = compute_fidelity_mps(psi0, psi_target, J_vec, U_vec, Delta_plus, s, config)
        grad_Delta[n] = (F_Delta_plus - F0) / ε
    end

    return grad_J, grad_U, grad_Delta, F0
end

"""
Compute stochastic gradients - only compute for a random subset of time steps.
Much faster for large n_steps.
"""
function compute_mps_gradients_stochastic(psi0::MPS, psi_target::MPS,
                                           J_vec::Vector{Float64}, U_vec::Vector{Float64},
                                           Δ_vec::Vector{Float64}, s, config::GRAPEConfig;
                                           ε::Float64=1e-5, batch_size::Int=20)
    n_steps = length(J_vec)

    grad_J = zeros(n_steps)
    grad_U = zeros(n_steps)
    grad_Delta = zeros(n_steps)

    # Base fidelity
    F0 = compute_fidelity_mps(psi0, psi_target, J_vec, U_vec, Δ_vec, s, config)

    # Sample random subset of time steps
    indices = randperm(n_steps - 1)[1:min(batch_size, n_steps - 1)]

    # Compute gradients only for sampled indices
    for n in indices
        # J gradient
        J_plus = copy(J_vec)
        J_plus[n] += ε
        F_J_plus = compute_fidelity_mps(psi0, psi_target, J_plus, U_vec, Δ_vec, s, config)
        grad_J[n] = (F_J_plus - F0) / ε

        # U gradient
        U_plus = copy(U_vec)
        U_plus[n] += ε
        F_U_plus = compute_fidelity_mps(psi0, psi_target, J_vec, U_plus, Δ_vec, s, config)
        grad_U[n] = (F_U_plus - F0) / ε

        # Delta gradient
        Delta_plus = copy(Δ_vec)
        Delta_plus[n] += ε
        F_Delta_plus = compute_fidelity_mps(psi0, psi_target, J_vec, U_vec, Delta_plus, s, config)
        grad_Delta[n] = (F_Delta_plus - F0) / ε
    end

    # Scale gradients to account for sampling (approximate full gradient)
    scale = (n_steps - 1) / length(indices)
    grad_J .*= scale
    grad_U .*= scale
    grad_Delta .*= scale

    return grad_J, grad_U, grad_Delta, F0
end

# ============================================================================
# GRAPE optimization with L-BFGS
# ============================================================================

"""
GRAPE optimization with Trotter propagation using L-BFGS (MPS version).

Parameters:
- s: Site indices
- config: GRAPE configuration
- J0, U0, Delta0: Initial control pulses
- verbose: Print progress

Returns optimized control pulses and final fidelity.
"""
function grape_mps_optimize(s, config::GRAPEConfig;
                            J0=nothing, U0=nothing, Delta0=nothing,
                            verbose=true)
    n_steps = config.n_steps

    # Initialize controls
    J_ctrl = isnothing(J0) ? 0.1 * ones(n_steps) : copy(J0)
    U_ctrl = isnothing(U0) ? 0.01 * ones(n_steps) : copy(U0)
    Delta_ctrl = isnothing(Delta0) ? zeros(n_steps) : copy(Delta0)

    # Create initial and target states
    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    # Pack controls into single vector for optimizer: [J; U; Delta]
    function pack_controls(J, U, Delta)
        return vcat(J, U, Delta)
    end

    function unpack_controls(x)
        J = x[1:n_steps]
        U = x[n_steps+1:2*n_steps]
        Delta = x[2*n_steps+1:3*n_steps]
        return J, U, Delta
    end

    # Objective function: minimize infidelity (1 - fidelity)
    function objective(x)
        J, U, Delta = unpack_controls(x)
        fidelity = compute_fidelity_mps(psi0, psi_target, J, U, Delta, s, config)
        return 1.0 - fidelity  # Minimize infidelity
    end

    # Gradient function using numerical differentiation
    function gradient!(g, x)
        J, U, Delta = unpack_controls(x)

        # Compute gradients numerically
        grad_J, grad_U, grad_Delta, _ = compute_mps_gradients_numerical(
            psi0, psi_target, J, U, Delta, s, config)

        # Negate for infidelity gradient (we're minimizing 1-F)
        g[1:n_steps] .= -grad_J
        g[n_steps+1:2*n_steps] .= -grad_U
        g[2*n_steps+1:3*n_steps] .= -grad_Delta
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
        LBFGS(m=20),
        Optim.Options(
            iterations=config.max_iterations,
            g_tol=config.tolerance * 1e-2,
            f_reltol=config.tolerance * 1e-2,
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
# Analysis functions
# ============================================================================

"""
Analyze MPS state to check if it's a NOON state.
"""
function analyze_noon_state(psi::MPS, s, config::GRAPEConfig)
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
    prob_left = abs2(inner(psi_left, psi))
    prob_right = abs2(inner(psi_right, psi))

    # NOON state fidelity
    target = make_noon_state(s, config)
    fidelity = abs2(inner(target, psi))

    return (prob_left=prob_left, prob_right=prob_right, fidelity=fidelity)
end

# ============================================================================
# Main execution: Run GRAPE optimization with MPS
# ============================================================================

"""
Gradient descent with momentum and adaptive learning rate.
Uses stochastic gradients for faster computation with many time steps.
"""
function grape_mps_simple(s, config::GRAPEConfig;
                          J0=nothing, U0=nothing, Delta0=nothing,
                          learning_rate=0.1, batch_size=20, verbose=true)
    n_steps = config.n_steps

    # Initialize controls
    J = isnothing(J0) ? 0.5 * ones(n_steps) : copy(J0)
    U = isnothing(U0) ? 0.1 * ones(n_steps) : copy(U0)
    Delta = isnothing(Delta0) ? zeros(n_steps) : copy(Delta0)

    # Create initial and target states
    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    best_fidelity = 0.0
    best_J, best_U, best_Delta = copy(J), copy(U), copy(Delta)

    # Momentum terms
    momentum = 0.9
    v_J = zeros(n_steps)
    v_U = zeros(n_steps)
    v_Delta = zeros(n_steps)

    lr = learning_rate
    prev_fidelity = 0.0

    # Use stochastic gradients if n_steps is large
    use_stochastic = (n_steps > 30)
    if verbose && use_stochastic
        println("Using stochastic gradients with batch_size=$batch_size")
    end

    for iter in 1:config.max_iterations
        # Compute gradients (stochastic for large n_steps)
        if use_stochastic
            grad_J, grad_U, grad_Delta, fidelity = compute_mps_gradients_stochastic(
                psi0, psi_target, J, U, Delta, s, config; batch_size=batch_size)
        else
            grad_J, grad_U, grad_Delta, fidelity = compute_mps_gradients_numerical(
                psi0, psi_target, J, U, Delta, s, config)
        end

        # Track best
        if fidelity > best_fidelity
            best_fidelity = fidelity
            best_J, best_U, best_Delta = copy(J), copy(U), copy(Delta)
        end

        # Print progress
        if verbose && (iter == 1 || iter % 10 == 0)
            @printf("Iter %3d: fidelity = %.6f (best = %.6f, lr = %.4f)\n",
                    iter, fidelity, best_fidelity, lr)
        end

        # Check convergence
        if best_fidelity > 1.0 - config.tolerance
            if verbose
                println("Converged!")
            end
            break
        end

        # Adaptive learning rate
        if iter > 1
            if fidelity < prev_fidelity - 0.05
                lr *= 0.7  # Reduce if fidelity dropped
            elseif fidelity > prev_fidelity
                lr = min(lr * 1.02, learning_rate)  # Slowly increase
            end
        end
        prev_fidelity = fidelity

        # Momentum update
        v_J = momentum .* v_J .+ lr .* grad_J
        v_U = momentum .* v_U .+ lr .* grad_U
        v_Delta = momentum .* v_Delta .+ lr .* grad_Delta

        J .+= v_J
        U .+= v_U
        Delta .+= v_Delta

        # Keep J positive
        J .= max.(J, 0.01)
    end

    if verbose
        println("Best fidelity: $best_fidelity")
    end

    return best_J, best_U, best_Delta, best_fidelity
end

function main()
    # System parameters
    N = 3  # Number of bosons
    k = 3  # Number of sites
    T = 10.0
    num_steps = 101  # More steps for finer time resolution
    dt = T / (num_steps - 1)

    println("="^60)
    println("GRAPE Optimal Control (MPS) for Bose-Hubbard Model")
    println("="^60)
    println("N = $N bosons, k = $k sites")
    println("Using 2nd-order Trotter with MPS representation")
    println("Time: T = $T, steps = $num_steps, dt = $dt")
    println("Note: Using numerical gradients with simple gradient descent")

    # Configuration
    config = GRAPEConfig(
        k,          # n_sites
        N,          # n_particles
        num_steps,  # n_steps
        dt,         # dt
        1e-8,       # cutoff
        500,        # max_iterations (more for stochastic gradients)
        1e-3        # tolerance
    )

    # Create site indices
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

    # Initial and target states
    psi0 = make_initial_state(s, config)
    psi_target = make_noon_state(s, config)

    println("\nInitial state: |$(config.n_particles),0,0⟩")
    println("Target state: NOON state (|$(config.n_particles),0,0⟩ + |0,0,$(config.n_particles)⟩)/√2")
    println("Initial fidelity: $(abs2(inner(psi_target, psi0)))")

    # Initial guess for controls
    println("\nStarting GRAPE optimization with MPS...")
    println("-"^60)

    Random.seed!(42)
    t_arr = collect(range(0, T, length=num_steps))
    J0 = 1.0 .+ 0.3 * sin.(2π * t_arr / T) .+ 0.1 * randn(num_steps)
    U0 = 0.1 .+ 0.05 * cos.(2π * t_arr / T) .+ 0.02 * randn(num_steps)
    Delta0 = 0.3 * sin.(4π * t_arr / T) .+ 0.05 * randn(num_steps)

    # Run GRAPE optimization with stochastic gradient descent + momentum
    J_opt, U_opt, Delta_opt, final_fidelity = grape_mps_simple(
        s, config;
        J0=J0, U0=U0, Delta0=Delta0,
        learning_rate=0.1,
        batch_size=20,  # Compute gradients for 20 random time steps per iteration
        verbose=true
    )

    println("-"^60)
    @printf("Final fidelity: %.6f\n", final_fidelity)

    # Final state analysis
    psi_final = forward_propagate(psi0, J_opt, U_opt, Delta_opt, s, config)
    result = analyze_noon_state(psi_final, s, config)

    println("\nFinal state analysis:")
    @printf("  |⟨ψ_target|ψ_final⟩|² = %.6f\n", result.fidelity)
    @printf("  |⟨%d,0,0|ψ_final⟩|² = %.6f\n", config.n_particles, result.prob_left)
    @printf("  |⟨0,0,%d|ψ_final⟩|² = %.6f\n", config.n_particles, result.prob_right)

    # Save results
    println("\nSaving results...")
    open("J_opt_mps.txt", "w") do f
        for val in J_opt
            println(f, val)
        end
    end
    open("U_opt_mps.txt", "w") do f
        for val in U_opt
            println(f, val)
        end
    end
    open("Delta_opt_mps.txt", "w") do f
        for val in Delta_opt
            println(f, val)
        end
    end
    println("Control pulses saved to J_opt_mps.txt, U_opt_mps.txt, Delta_opt_mps.txt")

    return J_opt, U_opt, Delta_opt, final_fidelity
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
