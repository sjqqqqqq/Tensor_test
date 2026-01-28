using Pkg
# Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using DelimitedFiles


struct SimulationConfig
    n_sites::Int
    n_particles::Int
    time_step::Float64
    final_time::Float64
    cutoff::Float64
    maxdim::Int
end

# Initialize simulation configuration
const CONFIG = SimulationConfig(
    3,              # n_sites
    20,             # n_particles
    0.02,           # time_step
    10.0,           # final_time
    1E-8,           # cutoff
    50,            # maxdim for TDVP
)


function build_hamiltonian_mpo(J::Float64, U::Float64, Δ::Float64, s, config::SimulationConfig)
    center = (config.n_sites + 1) / 2

    os = OpSum()

    # Onsite terms: U * N(N-1) + Δ * (j - center) * N
    for j in 1:config.n_sites
        # Interaction: U * n_j * (n_j - 1) = U * N^2 - U * N
        os += U, "N * N", j
        os += -U, "N", j
        # Tilt potential
        os += Δ * (j - center), "N", j
    end

    # Hopping terms: J * (a†_j a_{j+1} + h.c.)
    for j in 1:(config.n_sites - 1)
        os += J, "Adag", j, "A", j + 1
        os += J, "A", j, "Adag", j + 1
    end

    H = MPO(os, s)
    return H
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

        # println("t = $t\tQFI = $(round(QFI, digits=2))\tJ = $(round(J_t, digits=3))\tU = $(round(U_t, digits=3))\tΔ = $(round(Δ_t, digits=3))")

        # Build Hamiltonian MPO for current parameters
        H = build_hamiltonian_mpo(J_t, U_t, Δ_t, s, config)

        # Time evolve using TDVP (imaginary time step for real-time evolution: -im * dt)
        psi = tdvp(H, -im * config.time_step, psi;
                   cutoff=config.cutoff,
                   maxdim=config.maxdim,
                   normalize=true,
                   outputlevel=0)
    end

    return QFI_history, J_history, U_history, Δ_history
end


QFI_data, J_data, U_data, Δ_data = run_simulation(CONFIG);