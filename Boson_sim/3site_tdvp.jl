using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 3
    N_particles = 20

    a = [0.07747897282755545, -0.563632017585953, 0.47017428975720826, -0.2745107053189222, 0.13071778117587068]
    b = [1.0471264153979678, 0.39289819529909775, -0.0758928824380212, -0.08082406801993186, -0.41591786297140004]

    # Time-dependent parameters
    J(t) = 1.0 + 0.0*t
    U(t) = a[1] + a[2]*t + a[3]*t^2 + a[4]*t^3 + a[5]*t^4
    Δ(t) = b[1] + b[2]*t + b[3]*t^2 + b[4]*t^3 + b[5]*t^4

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
    global Q_list = []
    global bond_dims = []
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

        QFI = real(4*((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))
        push!(Q_list, QFI)

        # Compute bond dimension (maximum bond dimension across all bonds)
        max_bond_dim = maxlinkdim(psi)
        push!(bond_dims, max_bond_dim)

        J_t, U_t, Δ_t = J(t), U(t), Δ(t)

        # Build Hamiltonian using pre-computed MPO components
        H = make_H(J_t, U_t, Δ_t)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, maxdim=21, normalize=true, nsite=2)
    end

    println("Final QFI: ", Q_list[end])
    println("Final bond dimension: ", bond_dims[end])
end
