using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 5
    N_particles = 10

    # Time-dependent parameters
    J(t) = 1.0 + 0.5*sin(0.5*t)
    U(t) = 1.0 * cos(0.5 * t)
    Δ(t) = -0.1 * t

    # Time evolution parameters
    ts = 0.01
    tf = 5.0
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
    psi = MPS(s, ["$N_particles", "0", "0", "0", "0"])

    println("Time\tSite1\tSite2\tSite3\tSite4\tSite5\tTotal\tQFI")

    # Time evolution using TDVP
    global List = []
    for t in 0:ts:tf
        n1, n2, n3, n4, n5 = expect(psi, "N"; sites=1), expect(psi, "N"; sites=2), expect(psi, "N"; sites=3), expect(psi, "N"; sites=4), expect(psi, "N"; sites=5)
        N1, N2, N3, N4, N5 = expect(psi, "N * N"; sites=1), expect(psi, "N * N"; sites=2), expect(psi, "N * N"; sites=3), expect(psi, "N * N"; sites=4), expect(psi, "N * N"; sites=5)

        # Compute <n1*n3> using correlation_matrix
        n1n2 = correlation_matrix(psi, "N", "N")[1, 2]
        n1n4 = correlation_matrix(psi, "N", "N")[1, 4]
        n1n5 = correlation_matrix(psi, "N", "N")[1, 5]
        n2n4 = correlation_matrix(psi, "N", "N")[2, 4]
        n2n5 = correlation_matrix(psi, "N", "N")[2, 5]
        n4n5 = correlation_matrix(psi, "N", "N")[4, 5]

        QFI = 4*((4*N1 + 4*n1n2 - 4*n1n4 - 8*n1n5 + N2 - 2*n2n4 - 4*n2n5 + N4 + 4*n4n5 + 4*N5) - (-2n1 + -n2 + n4 + 2*n5)^2)
        J_t, U_t, Δ_t= J(t), U(t), Δ(t)
        println("$t\t$(round(n1, digits=2))\t$(round(n2, digits=2))\t$(round(n3, digits=2))\t$(round(n4, digits=2))\t$(round(n5, digits=2))\t$(round(n1+n2+n3+n4+n5, digits=2))\t$(round(QFI, digits=2))")

        # Build Hamiltonian at current time
        H = make_H(J_t, U_t, Δ_t, s)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, normalize=true, nsite=2)
        push!(List, QFI)
    end
end