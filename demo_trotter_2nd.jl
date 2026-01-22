using Pkg
# Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using DelimitedFiles
using Combinatorics
using LinearAlgebra

include("BH_basis.jl")


struct SimulationConfig
    n_sites::Int
    n_particles::Int
    time_step::Float64
    final_time::Float64
    cutoff::Float64
end

# Initialize simulation configuration
const CONFIG = SimulationConfig(
    3,              # n_sites
    10,            # n_particles
    0.01,            # time_step
    10.0,           # final_time
    1E-8,           # cutoff
)


function build_onsite_hamiltonian(site_index::Int, U::Float64, Δ::Float64, s, config::SimulationConfig)
    center = (config.n_sites + 1) / 2
    h = U * op("N * N", s[site_index]) + U * (-1.0) * op("N", s[site_index])
    h += Δ * (site_index - center) * op("N", s[site_index])
    return h
end


function build_hopping_hamiltonian(site_index::Int, J::Float64, s)
    h = -J * op("Adag", s[site_index]) * op("A", s[site_index + 1])
    h += -J * op("A", s[site_index]) * op("Adag", s[site_index + 1])
    return h
end


function make_trotter_gates_2nd(J::Float64, U::Float64, Δ::Float64, dt::Float64, s, config::SimulationConfig)
    gates = ITensor[]

    # First half of one-site gates
    for j in 1:config.n_sites
        hj = build_onsite_hamiltonian(j, U, Δ, s, config)
        Gj = exp(-im * dt/2 * hj)
        push!(gates, Gj)
    end

    # Two-site hopping gates
    for j in 1:(config.n_sites - 1)
        hj = build_hopping_hamiltonian(j, J, s)
        Gj = exp(-im * dt * hj)
        push!(gates, Gj)
    end

    # Second half of one-site gates
    for j in 1:config.n_sites
        hj = build_onsite_hamiltonian(j, U, Δ, s, config)
        Gj = exp(-im * dt/2 * hj)
        push!(gates, Gj)
    end

    return gates
end


function compute_qfi(psi, s, cutoff::Float64)
    # Batch expectation value calculations for efficiency
    n_vals = expect(psi, "N")
    n1, n3 = n_vals[1], n_vals[3]

    N_vals = expect(psi, "N * N")
    N1, N3 = N_vals[1], N_vals[3]

    # Compute ⟨n₁*n₃⟩ directly using inner product
    Op1 = op("N", s[1])
    Op3 = op("N", s[3])
    psi_temp = apply(Op3, psi; cutoff)
    n1n3 = real(inner(apply(Op1, psi; cutoff), psi_temp))

    # QFI formula
    QFI = real(4 * ((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))
    return QFI
end


function run_simulation(config::SimulationConfig)
    # Create site indices with quantum number conservation
    s = siteinds("Boson", config.n_sites; dim=config.n_particles + 1, conserve_qns=true)

    # Initialize state: all particles in first site
    psi = MPS(s, ["$(config.n_particles)", "0", "0"])

    # Storage for observables and parameters
    QFI_history = Float64[]
    J_history = Float64[]
    U_history = Float64[]
    Δ_history = Float64[]

    # Time evolution loop
    for t in 0:config.time_step:config.final_time
        # Compute and store QFI
        QFI = compute_qfi(psi, s, config.cutoff)
        push!(QFI_history, QFI)

        # Sample random parameters for this time step
        J_t = rand()
        U_t = 2 * rand() - 1
        Δ_t = 2 * rand() - 1

        push!(J_history, J_t)
        push!(U_history, U_t)
        push!(Δ_history, Δ_t)

        println("t = $t\tQFI = $(round(QFI, digits=2))\tJ = $(round(J_t, digits=3))\tU = $(round(U_t, digits=3))\tΔ = $(round(Δ_t, digits=3))")

        # Build and apply 2nd order Trotter gates
        gates = make_trotter_gates_2nd(J_t, U_t, Δ_t, config.time_step, s, config)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
    end

    return QFI_history, J_history, U_history, Δ_history, psi
end


QFI_data, J_data, U_data, Δ_data, final_psi = run_simulation(CONFIG)


function mps_to_statevector(psi::MPS, s, config::SimulationConfig)
    # Generate basis for N particles on L sites
    Basis, Ind = make_basis(config.n_particles, config.n_sites)
    dim = length(Basis)

    # Initialize state vector
    state_vector = zeros(ComplexF64, dim)

    # For each basis state, compute the overlap ⟨basis_state|psi⟩
    for (i, occ) in enumerate(Basis)
        # Create product state string for this occupation: ["n1", "n2", "n3"]
        state_str = [string(Int(n)) for n in occ]

        # Create MPS for this basis state
        basis_mps = MPS(s, state_str)

        # Compute overlap
        state_vector[i] = inner(basis_mps, psi)
    end

    return state_vector, Basis, Ind
end

# Convert final_psi to state vector (use siteinds from the MPS itself)
s = siteinds(final_psi)
state_vec, Basis, Ind = mps_to_statevector(final_psi, s, CONFIG)
psi_re, psi_im = real(state_vec), imag(state_vec)