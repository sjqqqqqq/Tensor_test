using Pkg
Pkg.activate(".")

using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 3
    N_particles = 100

    # Time evolution parameters
    ts = 0.01
    tf = 1.0
    cutoff = 1E-8

    # Create site indices
    s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

    # PRE-CONSTRUCT ALL OPERATORS ONCE (Major optimization)
    # One-site operators
    N_ops = [op("N", s[j]) for j in 1:N_sites]
    N2_ops = [op("N * N", s[j]) for j in 1:N_sites]

    # Two-site hopping operators
    Adag_ops = [op("Adag", s[j]) for j in 1:N_sites-1]
    A_ops = [op("A", s[j]) for j in 1:N_sites-1]
    A_next_ops = [op("A", s[j+1]) for j in 1:N_sites-1]
    Adag_next_ops = [op("Adag", s[j+1]) for j in 1:N_sites-1]

    # Pre-compute position factors
    pos_factors = [(j - (N_sites+1)/2) for j in 1:N_sites]

    # Helper function to build Trotter gates using pre-constructed operators
    function make_trotter_gates(J_val, U_val, Δ_val, dt, N_ops, N2_ops, Adag_ops, A_ops, A_next_ops, Adag_next_ops, pos_factors)
        gates = ITensor[]

        # One-site gates: U(t) * (N^2 - N) + Δ(t) * (j - (N_sites+1)/2) * N (first half)
        for j in 1:N_sites
            hj = U_val * N2_ops[j] + U_val * (-1.0) * N_ops[j]
            hj += Δ_val * pos_factors[j] * N_ops[j]
            Gj = exp(-im * dt/2 * hj)
            push!(gates, Gj)
        end

        # Two-site hopping gates: -J(t) * (adag_j * a_{j+1} + a_j * adag_{j+1})
        for j in 1:N_sites-1
            hj = -J_val * Adag_ops[j] * A_next_ops[j]
            hj += -J_val * A_ops[j] * Adag_next_ops[j]
            Gj = exp(-im * dt * hj)
            push!(gates, Gj)
        end

        # One-site gates again (second half of symmetric Trotter)
        for j in 1:N_sites
            hj = U_val * N2_ops[j] + U_val * (-1.0) * N_ops[j]
            hj += Δ_val * pos_factors[j] * N_ops[j]
            Gj = exp(-im * dt/2 * hj)
            push!(gates, Gj)
        end

        return gates
    end

    # Initial state
    psi = MPS(s, ["$N_particles", "0", "0"])

    # Pre-allocate arrays
    n_steps = Int(tf / ts) + 1
    Q_list = Vector{Float64}(undef, n_steps)
    Jt = Vector{Float64}(undef, n_steps)
    Ut = Vector{Float64}(undef, n_steps)
    Δt = Vector{Float64}(undef, n_steps)

    # Time evolution using Trotterization
    step = 1
    for t in 0:ts:tf
        # Batch expectation value calculations
        n_vals = expect(psi, "N")
        n1, n2, n3 = n_vals[1], n_vals[2], n_vals[3]

        N_vals = expect(psi, "N * N")
        N1, N2, N3 = N_vals[1], N_vals[2], N_vals[3]

        # Compute <n1*n3> using pre-constructed operators
        psi_temp = apply(N_ops[3], psi; cutoff)
        n1n3 = real(inner(apply(N_ops[1], psi; cutoff), psi_temp))

        QFI = real(4*((N1 + N3 - 2*n1n3) - (-n1 + n3)^2))
        Q_list[step] = QFI

        J_t, U_t, Δ_t = rand(), 2*rand()-1, 2*rand()-1
        Jt[step] = J_t
        Ut[step] = U_t
        Δt[step] = Δ_t

        # Build and apply Trotter gates using pre-constructed operators
        gates = make_trotter_gates(J_t, U_t, Δ_t, ts, N_ops, N2_ops, Adag_ops, A_ops, A_next_ops, Adag_next_ops, pos_factors)
        psi = apply(gates, psi; cutoff)
        normalize!(psi)

        step += 1
    end

    println("Final QFI: ", Q_list[end])

    # Write results
    using DelimitedFiles
    cd("Data")
    writedlm("QFI_MC_2nd_optimized.txt", Q_list)
    writedlm("J_MC_2nd_optimized.txt", Jt)
    writedlm("U_MC_2nd_optimized.txt", Ut)
    writedlm("Delta_MC_2nd_optimized.txt", Δt)
end
