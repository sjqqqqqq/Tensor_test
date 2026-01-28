using ITensors, ITensorMPS


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

    # Store initial state
    push!(psi_history, copy(psi))

    # Time evolution loop
    for t in 0:config.time_step:config.final_time

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
    end

    return psi_history, J_history, U_history, Δ_history
end

function backward_propagate(J_history::Vector{Float64}, U_history::Vector{Float64},
                            Δ_history::Vector{Float64}, config::SimulationConfig)
    # Create site indices with quantum number conservation
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

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


psi_data, J_data, U_data, Δ_data = forward_propagate(CONFIG)
chi_data = backward_propagate(J_data, U_data, Δ_data, CONFIG)
