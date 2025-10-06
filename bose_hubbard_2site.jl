using ITensors, ITensorMPS

# 2-site Bose Hubbard Model with 100 atoms total
let
    # Parameters
    N_sites = 2                    # Number of sites
    N_particles = 100              # Total number of bosons
    d = N_particles + 1            # Local Hilbert space dimension (0 to N_particles)

    # Bose Hubbard parameters
    t = 1.0                        # Hopping parameter
    U = 0.1                        # On-site interaction strength

    # Create site indices for bosons with conserved particle number
    s = siteinds("Boson", N_sites; dim=d, conserve_qns=true)

    # Build the Bose Hubbard Hamiltonian
    # H = -t * sum(b†_i * b_{i+1} + h.c.) + (U/2) * sum(n_i * (n_i - 1))

    os = OpSum()

    # Hopping terms (kinetic energy)
    for j in 1:(N_sites - 1)
        os += -t, "Adag", j, "A", j+1
        os += -t, "A", j, "Adag", j+1
    end

    # On-site interaction terms
    for j in 1:N_sites
        os += U/2, "N * N", j
        os += -U/2, "N", j
    end

    # Convert to MPO
    H = MPO(os, s)

    # Initialize state: all particles on first site
    # Create product state with occupation numbers
    states = ["$(N_particles)", "0"]
    psi0 = MPS(s, states)

    println("Initial state created")
    println("Site 1 occupation: ", expect(psi0, "N"; sites=1))
    println("Site 2 occupation: ", expect(psi0, "N"; sites=2))
    println("Total particles: ", sum(expect(psi0, "N")))
    println()
end
