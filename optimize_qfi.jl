using Pkg
Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using Evolutionary

# System parameters
const N_SITES = 3
const N_PARTICLES = 20

# Time evolution parameters
const TS = 0.01
const TF = 1.0
const CUTOFF = 1E-8


# Helper function to build time-dependent Hamiltonian MPO
function make_H(J_val, U_val, Δ_val, s)
    os = OpSum()
    # Hopping terms
    for j in 1:N_SITES-1
        os += -J_val, "Adag", j, "A", j+1
        os += -J_val, "A", j, "Adag", j+1
    end
    # On-site interaction terms
    for j in 1:N_SITES
        os += U_val, "N * N", j
        os += -U_val, "N", j
    end
    # Tilt terms
    for j in 1:N_SITES
        os += (j-(N_SITES+1)/2)*Δ_val, "N", j
    end
    return MPO(os, s)
end

# Objective function to minimize (returns -QFI to maximize QFI)
function objective(x)

    # Time-dependent parameters
    J(t) = 1.0 + 0.0*t
    U(t) = x[1] + x[2]*t + x[3]*t^2 + x[4]*t^3 + x[5]*t^4
    Δ(t) = x[6] * x[7]*t + x[8]*t^2 + x[9]*t^3 + x[10]*t^4

    # Create site indices
    s = siteinds("Boson", N_SITES; dim=N_PARTICLES + 1, conserve_qns=true)

    # Initial state: all particles on first site
    psi = MPS(s, ["$N_PARTICLES", "0", "0"])

    # Time evolution using TDVP
    QFI = 0.0
    try
        for t in 0:TS:TF
            n1 = expect(psi, "N"; sites=1)
            n3 = expect(psi, "N"; sites=3)
            N1 = expect(psi, "N * N"; sites=1)
            N3 = expect(psi, "N * N"; sites=3)

            # Compute <n1*n3> using correlation_matrix
            n1n3 = correlation_matrix(psi, "N", "N")[1, 3]

            # Compute QFI
            QFI = real(4*((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))

            # Get current time-dependent parameters
            J_t, U_t, Δ_t = J(t), U(t), Δ(t)

            # Build Hamiltonian at current time
            H = make_H(J_t, U_t, Δ_t, s)

            # Evolve using TDVP for one time step
            psi = tdvp(H, -im * TS, psi; cutoff=CUTOFF, normalize=true, nsite=2)
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
