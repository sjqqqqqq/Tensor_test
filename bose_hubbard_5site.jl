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
    num_steps = Int(tf/ts) + 1
    global List = Vector{Float64}(undef, num_steps)

    for (step_idx, t) in enumerate(0:ts:tf)
        # Batch expect calls for "N" operator
        n_vals = expect(psi, "N")
        n1, n2, n3, n4, n5 = n_vals[1], n_vals[2], n_vals[3], n_vals[4], n_vals[5]

        # Batch expect calls for "N * N" operator
        N_vals = expect(psi, "N * N")
        N1, N2, N4, N5 = N_vals[1], N_vals[2], N_vals[4], N_vals[5]

        # Compute correlation matrix once and extract needed elements
        corr_mat = correlation_matrix(psi, "N", "N")
        n1n2, n1n4, n1n5 = corr_mat[1, 2], corr_mat[1, 4], corr_mat[1, 5]
        n2n4, n2n5, n4n5 = corr_mat[2, 4], corr_mat[2, 5], corr_mat[4, 5]

        QFI = 4*((4*N1 + 4*n1n2 - 4*n1n4 - 8*n1n5 + N2 - 2*n2n4 - 4*n2n5 + N4 + 4*n4n5 + 4*N5) - (-2n1 + -n2 + n4 + 2*n5)^2)

        # Pre-compute rounded values for printing
        n_total = n1 + n2 + n3 + n4 + n5
        println(t, '\t', round(n1, digits=2), '\t', round(n2, digits=2), '\t',
                round(n3, digits=2), '\t', round(n4, digits=2), '\t',
                round(n5, digits=2), '\t', round(n_total, digits=2), '\t',
                round(QFI, digits=2))

        # Build Hamiltonian at current time
        H = make_H(J(t), U(t), Δ(t), s)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, normalize=true, nsite=2)
        List[step_idx] = real(QFI)
    end
end
