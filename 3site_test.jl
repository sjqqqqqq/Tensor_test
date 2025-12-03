using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 3
    N_particles = 100
    b0 = 1.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    b4 = 0.0
    c0 = 1.0
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    c4 = 0.0

    # Time-dependent parameters
    J(t) = 1.0 + 0.0*t
    U(t) = b0 + b1*t + b2*t^2 + b3*t^3 + b4*t^4
    Δ(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4

    # Time evolution parameters
    ts = 0.01
    tf = 1.0
    cutoff = 1E-8

    # Create site indices
    s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

    # Pre-build MPO components (avoid rebuilding in each iteration)
    # Hopping MPO
    os_J = OpSum()
    for j in 1:N_sites-1
        os_J += -1.0, "Adag", j, "A", j+1
        os_J += -1.0, "A", j, "Adag", j+1
    end
    H_J = MPO(os_J, s)

    # Interaction MPO
    os_U = OpSum()
    for j in 1:N_sites
        os_U += 1.0, "N * N", j
        os_U += -1.0, "N", j
    end
    H_U = MPO(os_U, s)

    # Tilt MPO
    os_Δ = OpSum()
    for j in 1:N_sites
        os_Δ += (j-(N_sites+1)/2), "N", j
    end
    H_Δ = MPO(os_Δ, s)

    # Helper function to compute time-dependent Hamiltonian as linear combination
    function make_H(J_val, U_val, Δ_val)
        return J_val * H_J + U_val * H_U + Δ_val * H_Δ
    end

    # Initial state
    psi = MPS(s, ["$N_particles", "0", "0"])

    # Time evolution using TDVP
    QFI_final = 0.0
    for t in 0:ts:tf
        # Batch expectation value calculations for efficiency
        n_vals = expect(psi, "N")
        n1, n2, n3 = n_vals[1], n_vals[2], n_vals[3]

        N_vals = expect(psi, "N * N")
        N1, N2, N3 = N_vals[1], N_vals[2], N_vals[3]

        # Compute <n1*n3> directly using inner product (more efficient than correlation_matrix)
        Op1 = op("N", s[1])
        Op3 = op("N", s[3])
        psi_temp = apply(Op3, psi; cutoff)
        n1n3 = real(inner(apply(Op1, psi; cutoff), psi_temp))

        QFI_final = real(4*((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))

        J_t, U_t, Δ_t = J(t), U(t), Δ(t)

        # Build Hamiltonian using pre-computed MPO components
        H = make_H(J_t, U_t, Δ_t)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, normalize=true, nsite=2)
    end

    println("Final QFI: ", QFI_final)
end
