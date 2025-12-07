using Pkg
Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using Evolutionary

# System parameters
N_sites = 3
N_particles = 30

# Time evolution parameters
ts = 0.01
tf = 2.0
cutoff = 1E-8

# Create site indices
s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

# Pre-build MPO components (avoid rebuilding in each iteration)
# Hopping MPO
os_J = OpSum()
for j in 1:N_sites-1
    global os_J
    os_J += -1.0, "Adag", j, "A", j+1
    os_J += -1.0, "A", j, "Adag", j+1
end
H_J = MPO(os_J, s)

# Interaction MPO
os_U = OpSum()
for j in 1:N_sites
    global os_U
    os_U += 1.0, "N * N", j
    os_U += -1.0, "N", j
end
H_U = MPO(os_U, s)

# Tilt MPO
os_Δ = OpSum()
for j in 1:N_sites
    global os_Δ
    os_Δ += (j-(N_sites+1)/2), "N", j
end
H_Δ = MPO(os_Δ, s)


function make_H(J_val, U_val, Δ_val)
    return J_val * H_J + U_val * H_U + Δ_val * H_Δ
end

# Objective function to minimize (returns -QFI to maximize QFI)
function objective(x)

    # Time-dependent parameters
    J(t) = 1.0 + 0.0*t
    U(t) = x[1] + x[2]*t + x[3]*t^2 + x[4]*t^3 + x[5]*t^4
    Δ(t) = x[6] * x[7]*t + x[8]*t^2 + x[9]*t^3 + x[10]*t^4

    # Initial state: all particles on first site
    psi = MPS(s, ["$N_particles", "0", "0"])

    # Time evolution using TDVP
    QFI = 0.0
    try
        for t in 0:ts:tf
            n_vals = expect(psi, "N")
            n1, n2, n3 = n_vals[1], n_vals[2], n_vals[3]

            N_vals = expect(psi, "N * N")
            N1, N2, N3 = N_vals[1], N_vals[2], N_vals[3]

            # Compute <n1*n3> directly using inner product (more efficient than correlation_matrix)
            Op1 = op("N", s[1])
            Op3 = op("N", s[3])
            psi_temp = apply(Op3, psi; cutoff)
            n1n3 = real(inner(apply(Op1, psi; cutoff), psi_temp))

            # Compute QFI
            QFI = real(4*((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))

            # Get current time-dependent parameters
            J_t, U_t, Δ_t = J(t), U(t), Δ(t)

            # Build Hamiltonian at current time
            H = make_H(J_t, U_t, Δ_t)

            # Evolve using TDVP for one time step
            psi = tdvp(H, -im * ts, psi; cutoff, normalize=true, nsite=2)
        end
    catch e
        println("Error during simulation: $e")
        return Inf  # Return large value if simulation fails
    end

    # Return negative QFI (to minimize -QFI = maximize QFI)
    result = -QFI
    println("Parameters: a=$(x[1:5]), b=$(x[6:10]) => QFI=$QFI (objective=$result)")
    return result
end

# Main optimization
function main()
    println("Starting QFI optimization using CMAES")
    println("="^70)

    # Initial guess
    x0 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    # Run CMAES optimization
    result = Evolutionary.optimize(
        objective,
        x0,
        CMAES(),
        Evolutionary.Options(
            iterations=10,
            show_trace=true,
            store_trace=true
        )
    )

    println("\n" * "="^70)
    println("Optimization complete!")
    println("Maximum QFI = $(-result.minimum)")
    x = result.minimizer
    # println("optimal hopping J = 1/2 * (1.0 + sin($(x[3])*t + $(x[6])))")
    println("optimal interaction U = $(x[5])t⁴ + $(x[4])t³ + $(x[3])t² + $(x[2])t + $(x[1])")
    println("optimal tilt Δ = $(x[10])t⁴ + $(x[9])t³ + $(x[8])t² + $(x[7])t + $(x[6])")

    return result
end

main()
