using Pkg
# Pkg.activate(".")
Pkg.activate("Tensor_test")

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
    output_dir::String
    output_suffix::String
end

# Initialize simulation configuration
const CONFIG = SimulationConfig(
    3,              # n_sites
    100,            # n_particles
    0.1,            # time_step
    30.0,           # final_time
    1E-8,           # cutoff
    "Data",         # output_dir
    "MC_4th_30"     # output_suffix
)

# 4th order Suzuki-Trotter coefficients (Yoshida construction)
const YOSHIDA_P = 1.0 / (4.0 - 4.0^(1.0/3.0))  # ≈ 1.3512071919596578
const YOSHIDA_Q = 1.0 - 4.0 * YOSHIDA_P         # ≈ -1.7024143839193155
const TROTTER_COEFFS = [YOSHIDA_P, YOSHIDA_P, YOSHIDA_Q, YOSHIDA_P, YOSHIDA_P]

# ============================================================================
# Helper Functions
# ============================================================================

"""
    build_onsite_hamiltonian(site_index, U, Δ, s, config)

Constructs the on-site Hamiltonian for a given site including interaction and trap terms.
H_onsite = U * n * (n - 1) + Δ * (j - center) * n
"""
function build_onsite_hamiltonian(site_index::Int, U::Float64, Δ::Float64, s, config::SimulationConfig)
    center = (config.n_sites + 1) / 2
    h = U * op("N * N", s[site_index]) + U * (-1.0) * op("N", s[site_index])
    h += Δ * (site_index - center) * op("N", s[site_index])
    return h
end

"""
    build_hopping_hamiltonian(site_index, J, s)

Constructs the hopping Hamiltonian between neighboring sites.
H_hop = -J * (a†_j * a_{j+1} + a_j * a†_{j+1})
"""
function build_hopping_hamiltonian(site_index::Int, J::Float64, s)
    h = -J * op("Adag", s[site_index]) * op("A", s[site_index + 1])
    h += -J * op("A", s[site_index]) * op("Adag", s[site_index + 1])
    return h
end

"""
    apply_symmetric_step!(gates, J, U, Δ, τ, s, config)

Applies one symmetric 2nd-order Trotter step with time step τ.
Structure: exp(-iτA/2) * exp(-iτB) * exp(-iτA/2)
where A = on-site terms, B = hopping terms
"""
function apply_symmetric_step!(gates, J::Float64, U::Float64, Δ::Float64, τ::Float64, s, config::SimulationConfig)
    # First half of one-site gates: exp(-i*τ*A/2)
    for j in 1:config.n_sites
        hj = build_onsite_hamiltonian(j, U, Δ, s, config)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end

    # Two-site hopping gates: exp(-i*τ*B)
    for j in 1:(config.n_sites - 1)
        hj = build_hopping_hamiltonian(j, J, s)
        Gj = exp(-im * τ * hj)
        push!(gates, Gj)
    end

    # Second half of one-site gates: exp(-i*τ*A/2)
    for j in 1:config.n_sites
        hj = build_onsite_hamiltonian(j, U, Δ, s, config)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end
end

"""
    make_trotter_gates_4th(J, U, Δ, dt, s, config)

Builds 4th order Trotter gates for one time step using Yoshida construction.
"""
function make_trotter_gates_4th(J::Float64, U::Float64, Δ::Float64, dt::Float64, s, config::SimulationConfig)
    gates = ITensor[]

    # Apply 5 symmetric 2nd-order steps with Yoshida coefficients
    for coeff in TROTTER_COEFFS
        apply_symmetric_step!(gates, J, U, Δ, coeff * dt, s, config)
    end

    return gates
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

Runs the main time evolution simulation with random parameter sampling.
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
        J_t = rand()
        U_t = 2 * rand() - 1
        Δ_t = 2 * rand() - 1

        push!(J_history, J_t)
        push!(U_history, U_t)
        push!(Δ_history, Δ_t)

        println("t = $t\tQFI = $(round(QFI, digits=2))\tJ = $(round(J_t, digits=3))\tU = $(round(U_t, digits=3))\tΔ = $(round(Δ_t, digits=3))")

        # Build and apply 4th order Trotter gates
        gates = make_trotter_gates_4th(J_t, U_t, Δ_t, config.time_step, s, config)
        psi = apply(gates, psi; cutoff=config.cutoff)
        normalize!(psi)
    end

    return QFI_history, J_history, U_history, Δ_history
end

# ============================================================================
# Execute Simulation
# ============================================================================

QFI_data, J_data, U_data, Δ_data = run_simulation(CONFIG)

# save_simulation_data(CONFIG.output_dir, CONFIG.output_suffix,
#                      QFI_data, J_data, U_data, Δ_data)
