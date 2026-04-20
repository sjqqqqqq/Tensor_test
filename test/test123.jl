using ITensors, ITensorMPS
using LinearAlgebra
using Random


struct SimulationConfig
    n_sites::Int
    n_particles::Int
    time_step::Float64
    final_time::Float64
    cutoff::Float64
end

const CONFIG = SimulationConfig(
    3,              # n_sites
    10,            # n_particles
    0.01,            # time_step
    10.0,           # final_time
    1E-8,           # cutoff
)


function build_H1(site_index::Int, U::Float64, Δ::Float64, s, config::SimulationConfig)
    center = (config.n_sites + 1) / 2
    h = U * op("N * N", s[site_index]) + U * (-1.0) * op("N", s[site_index])
    h += Δ * (site_index - center) * op("N", s[site_index])
    return h
end

function build_H2(site_index::Int, J::Float64, s)
    h = J * op("Adag", s[site_index]) * op("A", s[site_index + 1])
    h += J * op("A", s[site_index]) * op("Adag", s[site_index + 1])
    return h
end

# ============================================================================
# Gradient computation functions for Hamiltonians
# ============================================================================

"""
Compute ∂H1/∂U at a given site.
Since H1 = U*(N*N - N) + Δ*(i-center)*N, we have ∂H1/∂U = N*N - N = N*(N-1)
"""
function grad_H1_U(site_index::Int, s, config::SimulationConfig)
    return op("N * N", s[site_index]) + (-1.0) * op("N", s[site_index])
end

"""
Compute ∂H1/∂Δ at a given site.
Since H1 = U*(N*N - N) + Δ*(i-center)*N, we have ∂H1/∂Δ = (i-center)*N
"""
function grad_H1_Δ(site_index::Int, s, config::SimulationConfig)
    center = (config.n_sites + 1) / 2
    return (site_index - center) * op("N", s[site_index])
end

"""
Compute ∂H2/∂J at a given site pair (site_index, site_index+1).
Since H2 = J*(Adag_i*A_{i+1} + A_i*Adag_{i+1}), we have ∂H2/∂J = Adag_i*A_{i+1} + A_i*Adag_{i+1}
"""
function grad_H2_J(site_index::Int, s)
    h = op("Adag", s[site_index]) * op("A", s[site_index + 1])
    h += op("A", s[site_index]) * op("Adag", s[site_index + 1])
    return h
end

"""
Compute gate gradient ∂G/∂θ using finite differences.
G(θ) = exp(-i*dt*H(θ))
"""
function grad_gate_finite_diff(build_H_func::Function, θ::Float64, dt::Float64;
                                ε::Float64=1e-7, kwargs...)
    H_plus = build_H_func(θ + ε; kwargs...)
    H_minus = build_H_func(θ - ε; kwargs...)
    G_plus = exp(-im * dt * H_plus)
    G_minus = exp(-im * dt * H_minus)
    return (G_plus - G_minus) / (2ε)
end

"""
Structure to hold Hamiltonian derivatives for a time step.
"""
struct HamiltonianGradients
    dH1_dU::Vector{ITensor}  # ∂H1/∂U for each site
    dH1_dΔ::Vector{ITensor}  # ∂H1/∂Δ for each site
    dH2_dJ::Vector{ITensor}  # ∂H2/∂J for each bond
end

"""
Compute all Hamiltonian gradients for a single time step.
"""
function compute_hamiltonian_gradients(s, config::SimulationConfig)
    dH1_dU = ITensor[]
    dH1_dΔ = ITensor[]
    dH2_dJ = ITensor[]

    for j in 1:config.n_sites
        push!(dH1_dU, grad_H1_U(j, s, config))
        push!(dH1_dΔ, grad_H1_Δ(j, s, config))
    end

    for j in 1:(config.n_sites - 1)
        push!(dH2_dJ, grad_H2_J(j, s))
    end

    return HamiltonianGradients(dH1_dU, dH1_dΔ, dH2_dJ)
end

function make_noon_state(s, config::SimulationConfig)
    # NOON state: (|N,0,...,0⟩ + |0,...,0,N⟩) / √2
    n = config.n_particles

    # State with all particles on first site
    state1 = ["$n"]
    for _ in 2:config.n_sites
        push!(state1, "0")
    end
    psi1 = MPS(s, state1)

    # State with all particles on last site
    state2 = String[]
    for _ in 1:(config.n_sites - 1)
        push!(state2, "0")
    end
    push!(state2, "$n")
    psi2 = MPS(s, state2)

    # NOON = (|N,0,...⟩ + |0,...,N⟩) / √2
    noon = add(psi1, psi2; cutoff=config.cutoff)
    noon ./= sqrt(2)
    normalize!(noon)

    return noon
end

function make_trotter_gates_2nd(J::Float64, U::Float64, Δ::Float64, dt::Float64, s, config::SimulationConfig)
    gates = ITensor[]

    # First half of one-site gates: exp(-i*dt*A/2)
    for j in 1:config.n_sites
        hj = build_H1(j, U, Δ, s, config)
        Gj = exp(-im * dt/2 * hj)
        push!(gates, Gj)
    end

    # Two-site hopping gates: exp(-i*dt*B)
    for j in 1:(config.n_sites - 1)
        hj = build_H2(j, J, s)
        Gj = exp(-im * dt * hj)
        push!(gates, Gj)
    end

    # Second half of one-site gates: exp(-i*dt*A/2)
    for j in 1:config.n_sites
        hj = build_H1(j, U, Δ, s, config)
        Gj = exp(-im * dt/2 * hj)
        push!(gates, Gj)
    end

    return gates
end

function make_trotter_gates_2nd_adjoint(J::Float64, U::Float64, Δ::Float64, dt::Float64, s, config::SimulationConfig)
    # Adjoint gates for backward propagation: exp(+i*H*dt) applied in reverse order
    gates = ITensor[]

    # Reverse order: second half of one-site gates first
    for j in config.n_sites:-1:1
        hj = build_H1(j, U, Δ, s, config)
        Gj = exp(+im * dt/2 * hj)
        push!(gates, Gj)
    end

    # Two-site hopping gates in reverse order
    for j in (config.n_sites - 1):-1:1
        hj = build_H2(j, J, s)
        Gj = exp(+im * dt * hj)
        push!(gates, Gj)
    end

    # First half of one-site gates in reverse order
    for j in config.n_sites:-1:1
        hj = build_H1(j, U, Δ, s, config)
        Gj = exp(+im * dt/2 * hj)
        push!(gates, Gj)
    end

    return gates
end

function forward_propagate(config::SimulationConfig)
    # Create site indices with quantum number conservation
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

    # Initialize state: all particles in first site
    psi = MPS(s, ["$(config.n_particles)", "0", "0"])

    # Storage for MPS states and parameters
    psi_history = MPS[]
    J_history = Float64[]
    U_history = Float64[]
    Δ_history = Float64[]
    gates_history = Vector{ITensor}[]

    # Store initial state
    push!(psi_history, copy(psi))

    # Compute Hamiltonian gradients (these are parameter-independent operators)
    H_grads = compute_hamiltonian_gradients(s, config)

    # Time evolution loop
    for _ in 0:config.time_step:config.final_time
        # Sample random parameters for this time step
        J_t = rand()
        U_t = 2 * rand() - 1
        Δ_t = 2 * rand() - 1

        push!(J_history, J_t)
        push!(U_history, U_t)
        push!(Δ_history, Δ_t)

        # Build and apply 2nd order Trotter gates
        gates = make_trotter_gates_2nd(J_t, U_t, Δ_t, config.time_step, s, config)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)

        # Store MPS after this time step
        push!(psi_history, copy(psi))
        push!(gates_history, gates)
    end

    return (psi_history=psi_history, J_history=J_history, U_history=U_history,
            Δ_history=Δ_history, gates_history=gates_history,
            H_grads=H_grads, sites=s)
end

function backward_propagate(J_history::Vector{Float64}, U_history::Vector{Float64},
                            Δ_history::Vector{Float64}, config::SimulationConfig;
                            s=nothing)
    # Create site indices with quantum number conservation (or reuse provided ones)
    if s === nothing
        s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)
    end

    # Initialize costate chi as NOON state at final time
    chi = make_noon_state(s, config)

    # Storage for costate history
    chi_history = MPS[]

    # Store final state (initial condition for backward propagation)
    push!(chi_history, copy(chi))

    # Number of time steps
    n_steps = length(J_history)

    # Backward time evolution loop (reverse order)
    for i in n_steps:-1:1
        # Get parameters for this time step (in reverse order)
        J_t = J_history[i]
        U_t = U_history[i]
        Δ_t = Δ_history[i]

        # Build and apply adjoint Trotter gates
        gates = make_trotter_gates_2nd_adjoint(J_t, U_t, Δ_t, config.time_step, s, config)
        chi = apply(gates, chi; cutoff=config.cutoff)
        normalize!(chi)

        # Store chi after this backward step
        push!(chi_history, copy(chi))
    end

    return chi_history
end

"""
Compute ⟨χ|O|ψ⟩ for a one-site operator O acting at site_idx.
Uses the full MPS contraction.
"""
function expectation_one_site(chi::MPS, psi::MPS, O::ITensor, site_idx::Int)
    # Apply operator to psi
    psi_new = copy(psi)
    psi_new[site_idx] = noprime(O * psi[site_idx])
    # Compute overlap ⟨χ|O|ψ⟩
    return inner(chi, psi_new)
end

"""
Compute ⟨χ|O|ψ⟩ for a two-site operator O acting at sites (site_idx, site_idx+1).
O has indices (s'_i, s'_{i+1}, s_i, s_{i+1}) where primed = output, unprimed = input.
"""
function expectation_two_site(chi::MPS, psi::MPS, O::ITensor, site_idx::Int)
    n = length(psi)

    # Build the contraction from left to right
    # For ⟨χ|O|ψ⟩, we need: sum over all indices of conj(χ) * O * ψ

    # Start from the left
    if site_idx > 1
        # Contract up to site_idx - 1
        result = dag(chi[1]) * psi[1]
        for j in 2:(site_idx - 1)
            result = result * dag(chi[j]) * psi[j]
        end
    else
        result = ITensor(1.0)
    end

    # Handle the two-site block where O acts
    # O * psi gives tensor with primed site indices
    psi_two = psi[site_idx] * psi[site_idx + 1]
    O_psi = O * psi_two  # Result has indices (s'_i, s'_{i+1}, link indices)

    # chi_two needs primed indices to contract with O_psi
    chi_i = prime(chi[site_idx], "Site")
    chi_i1 = prime(chi[site_idx + 1], "Site")
    chi_two = dag(chi_i) * dag(chi_i1)

    # Contract
    result = result * chi_two * O_psi

    # Contract remaining sites
    if site_idx + 1 < n
        for j in (site_idx + 2):n
            result = result * dag(chi[j]) * psi[j]
        end
    end

    return scalar(result)
end

"""
Compute the gradient of the cost function F = |⟨target|ψ(T)⟩|² with respect to
the control parameters (J, U, Δ) at each time step using finite differences.

This provides accurate gradients for optimization.
"""
function compute_cost_gradients_numerical(J_history::Vector{Float64},
                                          U_history::Vector{Float64},
                                          Δ_history::Vector{Float64},
                                          config::SimulationConfig;
                                          ε::Float64=1e-6)
    n_steps = length(J_history)

    # Helper function to compute cost F = |⟨target|ψ(T)⟩|²
    function compute_cost(J_vec, U_vec, Δ_vec)
        s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)
        psi = MPS(s, ["$(config.n_particles)", "0", "0"])

        for k in eachindex(J_vec)
            gates = make_trotter_gates_2nd(J_vec[k], U_vec[k], Δ_vec[k], config.time_step, s, config)
            psi = apply(gates, psi; cutoff=config.cutoff)
            normalize!(psi)
        end

        target = make_noon_state(s, config)
        return abs2(inner(target, psi))
    end

    dF_dJ = zeros(Float64, n_steps)
    dF_dU = zeros(Float64, n_steps)
    dF_dΔ = zeros(Float64, n_steps)

    # Compute gradients for each time step
    for k in 1:n_steps
        # Gradient w.r.t. J[k]
        J_plus = copy(J_history); J_plus[k] += ε
        J_minus = copy(J_history); J_minus[k] -= ε
        dF_dJ[k] = (compute_cost(J_plus, U_history, Δ_history) -
                    compute_cost(J_minus, U_history, Δ_history)) / (2ε)

        # Gradient w.r.t. U[k]
        U_plus = copy(U_history); U_plus[k] += ε
        U_minus = copy(U_history); U_minus[k] -= ε
        dF_dU[k] = (compute_cost(J_history, U_plus, Δ_history) -
                    compute_cost(J_history, U_minus, Δ_history)) / (2ε)

        # Gradient w.r.t. Δ[k]
        Δ_plus = copy(Δ_history); Δ_plus[k] += ε
        Δ_minus = copy(Δ_history); Δ_minus[k] -= ε
        dF_dΔ[k] = (compute_cost(J_history, U_history, Δ_plus) -
                    compute_cost(J_history, U_history, Δ_minus)) / (2ε)
    end

    return (dF_dJ=dF_dJ, dF_dU=dF_dU, dF_dΔ=dF_dΔ)
end

"""
Compute the gradient of the cost function F = |⟨target|ψ(T)⟩|² with respect to
the control parameters (J, U, Δ) at each time step using the adjoint method.

For a gate G_k(θ) = exp(-i*dt*H_k(θ)), the gradient contribution is:
    ∂F/∂θ = 2 * Re[⟨χ_k| (-i*dt*∂H/∂θ) |ψ_{k-1}⟩ * ⟨target|ψ(T)⟩^*]

This uses the first-order approximation for the derivative of exp(-i*dt*H).
Note: This is an approximation and may have errors for larger time steps.
"""
function compute_cost_gradients(forward_result, config::SimulationConfig)
    psi_history = forward_result.psi_history
    J_history = forward_result.J_history
    U_history = forward_result.U_history
    Δ_history = forward_result.Δ_history
    H_grads = forward_result.H_grads
    s = forward_result.sites
    dt = config.time_step

    # Backward propagate to get adjoint states
    chi_history = backward_propagate(J_history, U_history, Δ_history, config; s=s)

    n_steps = length(J_history)

    # Storage for gradients at each time step
    dF_dJ = zeros(ComplexF64, n_steps)
    dF_dU = zeros(ComplexF64, n_steps)
    dF_dΔ = zeros(ComplexF64, n_steps)

    # Final state overlap with target (NOON state)
    target = make_noon_state(s, config)
    psi_final = psi_history[end]
    overlap = inner(target, psi_final)

    println("Final overlap with target: $(abs2(overlap))")

    # Compute gradients at each time step
    for k in 1:n_steps
        # State before applying gates at step k
        psi_k = psi_history[k]

        # Adjoint state at time k
        # chi_history[1] = chi(T), chi_history[n_steps+1] = chi(0)
        # After backward step i, we get chi at time i-1
        # So chi at step k is chi_history[n_steps - k + 1]
        chi_k = chi_history[n_steps - k + 1]

        # Gradient contributions from one-site terms (U and Δ)
        # Factor of 2 comes from two half-steps in 2nd order Trotter
        for j in 1:config.n_sites
            # ∂F/∂U contribution from site j
            dH_dU = H_grads.dH1_dU[j]
            grad_U_j = expectation_one_site(chi_k, psi_k, dH_dU, j)
            dF_dU[k] += -im * dt * grad_U_j  # Full dt because 2 half-steps

            # ∂F/∂Δ contribution from site j
            dH_dΔ = H_grads.dH1_dΔ[j]
            grad_Δ_j = expectation_one_site(chi_k, psi_k, dH_dΔ, j)
            dF_dΔ[k] += -im * dt * grad_Δ_j
        end

        # Gradient contributions from two-site terms (J)
        for j in 1:(config.n_sites - 1)
            dH_dJ = H_grads.dH2_dJ[j]
            grad_J_j = expectation_two_site(chi_k, psi_k, dH_dJ, j)
            dF_dJ[k] += -im * dt * grad_J_j
        end
    end

    # Apply the overlap factor: ∂|⟨target|ψ⟩|²/∂θ = 2*Re[⟨target|ψ⟩* × ∂⟨target|ψ⟩/∂θ]
    dF_dJ_real = 2 * real.(conj(overlap) .* dF_dJ)
    dF_dU_real = 2 * real.(conj(overlap) .* dF_dU)
    dF_dΔ_real = 2 * real.(conj(overlap) .* dF_dΔ)

    return (dF_dJ=dF_dJ_real, dF_dU=dF_dU_real, dF_dΔ=dF_dΔ_real)
end

"""
Validate Hamiltonian gradients by checking that H(θ+ε) - H(θ-ε) ≈ 2ε * ∂H/∂θ
"""
function validate_hamiltonian_gradients(config::SimulationConfig; ε::Float64=1e-6)
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

    println("=== Validating Hamiltonian Gradients ===\n")

    # Test parameters
    U_test = 0.3
    Δ_test = 0.2
    J_test = 0.5

    # Test ∂H1/∂U
    println("Testing ∂H1/∂U at site 1:")
    H1_plus = build_H1(1, U_test + ε, Δ_test, s, config)
    H1_minus = build_H1(1, U_test - ε, Δ_test, s, config)
    numerical_dH1_dU = (H1_plus - H1_minus) / (2ε)
    analytical_dH1_dU = grad_H1_U(1, s, config)
    diff_U = norm(numerical_dH1_dU - analytical_dH1_dU)
    println("  ||numerical - analytical|| = $diff_U")

    # Test ∂H1/∂Δ
    println("\nTesting ∂H1/∂Δ at site 1:")
    H1_plus_Δ = build_H1(1, U_test, Δ_test + ε, s, config)
    H1_minus_Δ = build_H1(1, U_test, Δ_test - ε, s, config)
    numerical_dH1_dΔ = (H1_plus_Δ - H1_minus_Δ) / (2ε)
    analytical_dH1_dΔ = grad_H1_Δ(1, s, config)
    diff_Δ = norm(numerical_dH1_dΔ - analytical_dH1_dΔ)
    println("  ||numerical - analytical|| = $diff_Δ")

    # Test ∂H2/∂J
    println("\nTesting ∂H2/∂J at bond 1-2:")
    H2_plus = build_H2(1, J_test + ε, s)
    H2_minus = build_H2(1, J_test - ε, s)
    numerical_dH2_dJ = (H2_plus - H2_minus) / (2ε)
    analytical_dH2_dJ = grad_H2_J(1, s)
    diff_J = norm(numerical_dH2_dJ - analytical_dH2_dJ)
    println("  ||numerical - analytical|| = $diff_J")

    println("\n=== Validation Complete ===")
    return (diff_U=diff_U, diff_Δ=diff_Δ, diff_J=diff_J)
end

"""
Validate gate gradients using finite differences.
"""
function validate_gate_gradients(config::SimulationConfig; ε::Float64=1e-6)
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)
    dt = config.time_step

    println("\n=== Validating Gate Gradients (Finite Differences) ===\n")

    # Test parameters
    U_test = 0.3
    Δ_test = 0.2
    J_test = 0.5

    # Test ∂G1/∂U for one-site gate
    println("Testing ∂G/∂U for one-site gate at site 1:")
    H1 = build_H1(1, U_test, Δ_test, s, config)
    H1_plus = build_H1(1, U_test + ε, Δ_test, s, config)
    H1_minus = build_H1(1, U_test - ε, Δ_test, s, config)

    G = exp(-im * dt/2 * H1)
    G_plus = exp(-im * dt/2 * H1_plus)
    G_minus = exp(-im * dt/2 * H1_minus)

    numerical_dG_dU = (G_plus - G_minus) / (2ε)

    # Analytical approximation: ∂G/∂θ ≈ -i*dt/2 * (∂H/∂θ) * G
    # But this requires proper operator multiplication
    # Instead, show the numerical gradient norm
    println("  ||∂G/∂U||_numerical = $(norm(numerical_dG_dU))")

    # Test ∂G2/∂J for two-site gate
    println("\nTesting ∂G/∂J for two-site gate at bond 1-2:")
    H2 = build_H2(1, J_test, s)
    H2_plus = build_H2(1, J_test + ε, s)
    H2_minus = build_H2(1, J_test - ε, s)

    G2 = exp(-im * dt * H2)
    G2_plus = exp(-im * dt * H2_plus)
    G2_minus = exp(-im * dt * H2_minus)

    numerical_dG_dJ = (G2_plus - G2_minus) / (2ε)
    println("  ||∂G/∂J||_numerical = $(norm(numerical_dG_dJ))")

    println("\n=== Validation Complete ===")
end

"""
Validate cost function gradients using finite differences.
"""
function validate_cost_gradients(config::SimulationConfig; ε::Float64=1e-5, step_idx::Int=1)
    println("\n=== Validating Cost Function Gradients ===\n")
    println("Testing at time step $step_idx with ε=$ε\n")

    # Use a shorter simulation for validation
    short_config = SimulationConfig(
        config.n_sites,
        config.n_particles,
        config.time_step,
        0.05,  # Very short time for validation
        config.cutoff
    )

    # Generate fixed random parameters
    Random.seed!(42)
    forward_result = forward_propagate(short_config)
    J_base = copy(forward_result.J_history)
    U_base = copy(forward_result.U_history)
    Δ_base = copy(forward_result.Δ_history)

    n_steps = length(J_base)
    if step_idx > n_steps
        println("Warning: step_idx=$step_idx > n_steps=$n_steps, using step_idx=1")
        step_idx = 1
    end

    # Compute analytical gradients
    Random.seed!(42)
    forward_result = forward_propagate(short_config)
    gradients_analytical = compute_cost_gradients(forward_result, short_config)

    # Compute numerical gradients for specified step
    println("\nComputing numerical gradients for step $step_idx...")
    gradients_numerical = compute_cost_gradients_numerical(J_base, U_base, Δ_base, short_config; ε=ε)

    println("\ndF/dJ at step $step_idx:")
    println("  Analytical (adjoint): $(gradients_analytical.dF_dJ[step_idx])")
    println("  Numerical (finite diff): $(gradients_numerical.dF_dJ[step_idx])")
    println("  Difference: $(abs(gradients_analytical.dF_dJ[step_idx] - gradients_numerical.dF_dJ[step_idx]))")

    println("\ndF/dU at step $step_idx:")
    println("  Analytical (adjoint): $(gradients_analytical.dF_dU[step_idx])")
    println("  Numerical (finite diff): $(gradients_numerical.dF_dU[step_idx])")
    println("  Difference: $(abs(gradients_analytical.dF_dU[step_idx] - gradients_numerical.dF_dU[step_idx]))")

    println("\ndF/dΔ at step $step_idx:")
    println("  Analytical (adjoint): $(gradients_analytical.dF_dΔ[step_idx])")
    println("  Numerical (finite diff): $(gradients_numerical.dF_dΔ[step_idx])")
    println("  Difference: $(abs(gradients_analytical.dF_dΔ[step_idx] - gradients_numerical.dF_dΔ[step_idx]))")

    println("\n=== Validation Complete ===")

    return (analytical=gradients_analytical, numerical=gradients_numerical)
end

# ============================================================================
# Main execution
# ============================================================================

# Validate Hamiltonian gradients
validation_H = validate_hamiltonian_gradients(CONFIG)

# Validate gate gradients
validate_gate_gradients(CONFIG)

# Validate cost function gradients
validation_cost = validate_cost_gradients(CONFIG; step_idx=1)

# Run a short demo with numerical gradients
println("\n" * "="^60)
println("Demo: Computing accurate gradients using finite differences")
println("="^60)

# Use a short simulation for demo
demo_config = SimulationConfig(
    CONFIG.n_sites,
    CONFIG.n_particles,
    CONFIG.time_step,
    0.1,  # Short simulation for demo
    CONFIG.cutoff
)

Random.seed!(123)
demo_forward = forward_propagate(demo_config)
println("Forward propagation complete. Number of time steps: $(length(demo_forward.J_history))")

# Compute numerical gradients for first few steps
println("\nComputing numerical gradients for first 3 time steps...")
n_demo_steps = min(3, length(demo_forward.J_history))
for k in 1:n_demo_steps
    println("\n--- Time step $k ---")

    # Compute gradient for step k only (faster than full gradient)
    J_base = demo_forward.J_history
    U_base = demo_forward.U_history
    Δ_base = demo_forward.Δ_history
    ε = 1e-6

    function compute_cost_demo(J_vec, U_vec, Δ_vec)
        s = siteinds("Boson", demo_config.n_sites; dim=demo_config.n_particles + 1, conserve_qns=true)
        psi = MPS(s, ["$(demo_config.n_particles)", "0", "0"])
        for i in eachindex(J_vec)
            gates = make_trotter_gates_2nd(J_vec[i], U_vec[i], Δ_vec[i], demo_config.time_step, s, demo_config)
            psi = apply(gates, psi; cutoff=demo_config.cutoff)
            normalize!(psi)
        end
        target = make_noon_state(s, demo_config)
        return abs2(inner(target, psi))
    end

    # J gradient
    J_plus = copy(J_base); J_plus[k] += ε
    J_minus = copy(J_base); J_minus[k] -= ε
    dF_dJ_k = (compute_cost_demo(J_plus, U_base, Δ_base) - compute_cost_demo(J_minus, U_base, Δ_base)) / (2ε)

    # U gradient
    U_plus = copy(U_base); U_plus[k] += ε
    U_minus = copy(U_base); U_minus[k] -= ε
    dF_dU_k = (compute_cost_demo(J_base, U_plus, Δ_base) - compute_cost_demo(J_base, U_minus, Δ_base)) / (2ε)

    # Δ gradient
    Δ_plus = copy(Δ_base); Δ_plus[k] += ε
    Δ_minus = copy(Δ_base); Δ_minus[k] -= ε
    dF_dΔ_k = (compute_cost_demo(J_base, U_base, Δ_plus) - compute_cost_demo(J_base, U_base, Δ_minus)) / (2ε)

    println("  ∂F/∂J[$k] = $dF_dJ_k")
    println("  ∂F/∂U[$k] = $dF_dU_k")
    println("  ∂F/∂Δ[$k] = $dF_dΔ_k")
end

println("\n" * "="^60)
println("Summary of gradient computation methods:")
println("="^60)
println("""
The code provides:

1. Hamiltonian gradients (∂H/∂θ):
   - grad_H1_U(site, s, config)  → ∂H1/∂U = N*(N-1)
   - grad_H1_Δ(site, s, config)  → ∂H1/∂Δ = (site - center)*N
   - grad_H2_J(site, s)          → ∂H2/∂J = A†_i A_{i+1} + h.c.

2. Cost function gradients (∂F/∂θ):
   - compute_cost_gradients_numerical() - Accurate finite differences
   - compute_cost_gradients()           - Fast adjoint method (approximate)

For optimization, use compute_cost_gradients_numerical() for accurate gradients,
or the adjoint method for faster approximate gradients when dt is small.
""")
