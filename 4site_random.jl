using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 4
    N_particles = 100

    # Constant parameters
    J = 1.0
    U = 0.5
    Δ = 0.1

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
    psi = MPS(s, ["$N_particles", "0", "0", "0"])

    # Build Hamiltonian once (since parameters are constant)
    H = make_H(J, U, Δ, s)

    println("Time\tSite1\tSite2\tSite3\tSite4\tTotal\tQFI")

    # Time evolution using TDVP
    global List = []
    for t in 0:ts:tf
        n1, n2, n3, n4 = expect(psi, "N"; sites=1), expect(psi, "N"; sites=2), expect(psi, "N"; sites=3), expect(psi, "N"; sites=4)
        N1, N2, N3, N4 = expect(psi, "N * N"; sites=1), expect(psi, "N * N"; sites=2), expect(psi, "N * N"; sites=3), expect(psi, "N * N"; sites=4)

        # Compute <n1*n3> using correlation_matrix
        n1n2 = correlation_matrix(psi, "N", "N")[1, 2]
        n1n3 = correlation_matrix(psi, "N", "N")[1, 3]
        n1n4 = correlation_matrix(psi, "N", "N")[1, 4]
        n2n3 = correlation_matrix(psi, "N", "N")[2, 3]
        n2n4 = correlation_matrix(psi, "N", "N")[2, 4]
        n3n4 = correlation_matrix(psi, "N", "N")[3, 4]

        QFI = 4*((9*N1/4 + 3*n1n2/2 - 3*n1n3/2 - 9*n1n4/2 + N2/4 - n2n3/2 -3*n2n4/2 + N3/4 + 3*n3n4/2 + 9*N4/4) - (-1.5*n1 + -0.5*n2 + 0.5*n3 + 1.5*n4)^2)
        println("$t\t$(round(n1, digits=2))\t$(round(n2, digits=2))\t$(round(n3, digits=2))\t$(round(n4, digits=2))\t$(round(n1+n2+n3+n4, digits=2))\t$(round(QFI, digits=2))")

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, normalize=true, nsite=2)
        push!(List, QFI)
    end
end