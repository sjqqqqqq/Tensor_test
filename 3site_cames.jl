using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 3
    N_particles = 100
    a1 = 1.0
    a2 = 0.5
    ω1 = 2π
    ω2 = π
    ϕ1 = 0.0
    ϕ2 = π/2

    # Time-dependent parameters
    J(t) = 1.0 + 0.0*t
    U(t) = a1 * cos(ω1*t + ϕ1)
    Δ(t) = a2 * sin(ω2*t + ϕ2)

    # Time evolution parameters
    ts = 0.01
    tf = 1.0
    cutoff = 1E-8

    # Create site indices
    s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

    # Helper function to build time-dependent Hamiltonian MPO
    function make_H(J_val, U_val, Δ_val, s)
        os = OpSum()
        # Hopping terms
        for j in 1:N_sites-1
            os += -J_val, "Adag", j, "A", j+1
            os += -J_val, "A", j, "Adag", j+1
        end
        # On-site interaction terms
        for j in 1:N_sites
            os += U_val, "N * N", j
            os += -U_val, "N", j
        end
        # Tilt terms
        for j in 1:N_sites
            os += (j-(N_sites+1)/2)*Δ_val, "N", j
        end
        return MPO(os, s)
    end

    # Initial state
    psi = MPS(s, ["$N_particles", "0", "0"])

    # Time evolution using TDVP
    global QFI = 0
    for t in 0:ts:tf
        n1, n2, n3 = expect(psi, "N"; sites=1), expect(psi, "N"; sites=2), expect(psi, "N"; sites=3)
        N1, N2, N3 = expect(psi, "N * N"; sites=1), expect(psi, "N * N"; sites=2), expect(psi, "N * N"; sites=3)

        # Compute <n1*n3> using correlation_matrix
        n1n3 = correlation_matrix(psi, "N", "N")[1, 3]

        QFI = real(4*((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))
        J_t, U_t, Δ_t= J(t), U(t), Δ(t)

        # Build Hamiltonian at current time
        H = make_H(J_t, U_t, Δ_t, s)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, normalize=true, nsite=2)
    end
end
