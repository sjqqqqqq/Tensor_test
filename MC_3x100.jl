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

    # Helper function to build Trotter gates for one time step
    function make_trotter_gates(J_val, U_val, Δ_val, dt, s)
        gates = ITensor[]

        # One-site gates: U(t) * (N^2 - N) + Δ(t) * (j - (N_sites+1)/2) * N
        for j in 1:N_sites
            hj = U_val * op("N * N", s[j]) + U_val * (-1.0) * op("N", s[j])
            hj += Δ_val * (j - (N_sites+1)/2) * op("N", s[j])
            Gj = exp(-im * dt/2 * hj)
            push!(gates, Gj)
        end

        # Two-site hopping gates: -J(t) * (adag_j * a_{j+1} + a_j * adag_{j+1})
        for j in 1:N_sites-1
            hj = -J_val * op("Adag", s[j]) * op("A", s[j+1])
            hj += -J_val * op("A", s[j]) * op("Adag", s[j+1])
            Gj = exp(-im * dt * hj)
            push!(gates, Gj)
        end

        # One-site gates again (second half of symmetric Trotter)
        for j in 1:N_sites
            hj = U_val * op("N * N", s[j]) + U_val * (-1.0) * op("N", s[j])
            hj += Δ_val * (j - (N_sites+1)/2) * op("N", s[j])
            Gj = exp(-im * dt/2 * hj)
            push!(gates, Gj)
        end

        return gates
    end

    # Initial state
    psi = MPS(s, ["$N_particles", "0", "0"])

    # Time evolution using Trotterization
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

        # Build and apply Trotter gates
        gates = make_trotter_gates(J_t, U_t, Δ_t, ts, s)
        psi = apply(gates, psi; cutoff)
        normalize!(psi)
    end

    println("Final QFI: ", Q_list[end])
end

using DelimitedFiles

cd("Data")
writedlm("QFI_MC_10.txt", Q_list)
writedlm("J_MC_10.txt", Jt)
writedlm("U_MC_10.txt", Ut)
writedlm("Delta_MC_10.txt", Δt)
