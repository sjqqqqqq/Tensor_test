using Pkg
Pkg.activate(".")
# Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using DelimitedFiles

# ============================================================================
# Configuration Parameters
# ============================================================================
struct SimulationConfig
    n_sites::Int
    n_particles::Int
    time_step::Float64
    final_time::Float64
    cutoff::Float64
    maxdim::Int
    output_dir::String
    output_suffix::String
end

# Initialize simulation configuration
const CONFIG = SimulationConfig(
    3,              # n_sites
    20,            # n_particles
    0.01,            # time_step
    20.0,           # final_time
    1E-8,           # cutoff
    50,             # maxdim for TDVP
    "Data",         # output_dir
    "MC_tdvp_20"    # output_suffix
)

# ============================================================================
# Helper Functions
# ============================================================================

"""
    build_bose_hubbard_hamiltonian(J, U, Δ, s, config)

Constructs the Bose-Hubbard Hamiltonian as an OpSum for TDVP.
H = -J Σ (a†_j a_{j+1} + h.c.) + U Σ n_j(n_j - 1) + Δ Σ (j - center) n_j
"""
function build_bose_hubbard_hamiltonian(J::Float64, U::Float64, Δ::Float64, s, config::SimulationConfig)
    os = OpSum()
    center = (config.n_sites + 1) / 2

    # On-site interaction: U * n * (n - 1) = U * (N*N - N)
    for j in 1:config.n_sites
        os += U, "N * N", j
        os += -U, "N", j
    end

    # Trap potential: Δ * (j - center) * n
    for j in 1:config.n_sites
        coeff = Δ * (j - center)
        if abs(coeff) > 1e-15  # Skip zero coefficients
            os += coeff, "N", j
        end
    end

    # Hopping terms: -J * (a†_j a_{j+1} + a_j a†_{j+1})
    for j in 1:(config.n_sites - 1)
        os += -J, "Adag", j, "A", j + 1
        os += -J, "A", j, "Adag", j + 1
    end

    return MPO(os, s)
end

"""
    compute_qfi(psi, s, cutoff)

Computes the Quantum Fisher Information for the state psi.
QFI = 4 * (⟨ΔJ_z²⟩ - ⟨J_z⟩²) where J_z = (n₁ - n₃)/2
"""
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

"""
    save_simulation_data(output_dir, suffix, QFI_data, J_data, U_data, Δ_data)

Saves simulation results to text files in the specified directory.
"""
function save_simulation_data(output_dir::String, suffix::String,
                               QFI_data, J_data, U_data, Δ_data)
    isdir(output_dir) || mkdir(output_dir)
    cd(output_dir)

    writedlm("QFI_$(suffix).txt", QFI_data)
    writedlm("J_$(suffix).txt", J_data)
    writedlm("U_$(suffix).txt", U_data)
    writedlm("Delta_$(suffix).txt", Δ_data)
end

# ============================================================================
# Main Simulation
# ============================================================================

"""
    run_simulation(config)

Runs the main time evolution simulation with random parameter sampling using TDVP.
"""
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
        J_t = 2 * rand()
        U_t = 2 * (2 * rand() - 1)
        Δ_t = 2 * (2 * rand() - 1)

        push!(J_history, J_t)
        push!(U_history, U_t)
        push!(Δ_history, Δ_t)

        println("t = $t\tQFI = $(round(QFI, digits=2))\tJ = $(round(J_t, digits=3))\tU = $(round(U_t, digits=3))\tΔ = $(round(Δ_t, digits=3))")

        # Build Hamiltonian MPO for current parameters
        H = build_bose_hubbard_hamiltonian(J_t, U_t, Δ_t, s, config)

        # Time evolution using TDVP (real-time evolution: -i*t)
        psi = tdvp(
            H,
            -im * config.time_step,
            psi;
            time_step = -im * config.time_step,
            maxdim = config.maxdim,
            cutoff = config.cutoff,
            normalize = true,
            reverse_step = true,
            outputlevel = 0,
        )
    end

    return QFI_history, J_history, U_history, Δ_history
end

# ============================================================================
# Execute Simulation
# ============================================================================

QFI_data, J_data, U_data, Δ_data = run_simulation(CONFIG)

# save_simulation_data(CONFIG.output_dir, CONFIG.output_suffix,
#                      QFI_data, J_data, U_data, Δ_data)
