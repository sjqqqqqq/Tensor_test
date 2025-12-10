using Pkg
Pkg.activate("Tensor_test")

using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 3
    N_particles = 100

    # Time evolution parameters
    ts = 0.01
    tf = 10.0
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
    global Jt = []
    global Ut = []
    global Δt = []
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

        J_t, U_t, Δ_t = rand(), 2*rand()-1, 2*rand()-1
        push!(Jt, J_t)
        push!(Ut, U_t)
        push!(Δt, Δ_t)

        println("$t\t$(round(QFI, digits=2))")

        # Build Hamiltonian using pre-computed MPO components
        H = make_H(J_t, U_t, Δ_t)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * ts, psi; cutoff, maxdim=50, normalize=true, nsite=2)
    end

    println("Final QFI: ", Q_list[end])
end

using DelimitedFiles

cd("Data")
writedlm("QFI_rand_10.txt", Q_list)
writedlm("J_rand_10.txt", Jt)
writedlm("U_rand_10.txt", Ut)
writedlm("Delta_rand_10.txt", Δt)
