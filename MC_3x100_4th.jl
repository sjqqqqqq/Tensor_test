using Pkg
# Pkg.activate(".")
Pkg.activate("Tensor_test")

using ITensors, ITensorMPS

# System parameters
N_sites = 3
N_particles = 100

# Time evolution parameters
ts = 0.1
tf = 10.0
cutoff = 1E-8

# Create site indices
s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

# 4th order Suzuki-Trotter coefficients (Yoshida construction)
p = 1.0 / (4.0 - 4.0^(1.0/3.0))  # ≈ 1.3512071919596578
q = 1.0 - 4.0 * p                 # ≈ -1.7024143839193155
coeffs = [p, p, q, p, p]

# Helper function to apply one symmetric 2nd-order step with time step τ
function apply_symmetric_step!(gates, J_val, U_val, Δ_val, τ, s)
    # One-site gates (first half): exp(-i*τ*A/2)
    for j in 1:N_sites
        hj = U_val * op("N * N", s[j]) + U_val * (-1.0) * op("N", s[j])
        hj += Δ_val * (j - (N_sites+1)/2) * op("N", s[j])
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end

    # Two-site hopping gates: exp(-i*τ*B)
    for j in 1:N_sites-1
        hj = -J_val * op("Adag", s[j]) * op("A", s[j+1])
        hj += -J_val * op("A", s[j]) * op("Adag", s[j+1])
        Gj = exp(-im * τ * hj)
        push!(gates, Gj)
    end

    # One-site gates (second half): exp(-i*τ*A/2)
    for j in 1:N_sites
        hj = U_val * op("N * N", s[j]) + U_val * (-1.0) * op("N", s[j])
        hj += Δ_val * (j - (N_sites+1)/2) * op("N", s[j])
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end
end

# Helper function to build 4th order Trotter gates for one time step
function make_trotter_gates_4th(J_val, U_val, Δ_val, dt, s)
    gates = ITensor[]

    # Apply 5 symmetric 2nd-order steps with coefficients [p, p, q, p, p]
    for coeff in coeffs
        apply_symmetric_step!(gates, J_val, U_val, Δ_val, coeff * dt, s)
    end

    return gates
end


# Initial state
psi = MPS(s, ["$N_particles", "0", "0"])

# Time evolution using Trotterization
Q_list = []
Jt = []
Ut = []
Δt = []
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

    # Build and apply Trotter gates
    gates = make_trotter_gates_4th(J_t, U_t, Δ_t, ts, s)
    global psi = apply(gates, psi; cutoff)
    normalize!(psi)
end


using DelimitedFiles

isdir("Data") || mkdir("Data")
cd("Data")
writedlm("QFI_MC_4th_10.txt", Q_list)
writedlm("J_MC_4th_10.txt", Jt)
writedlm("U_MC_4th_10.txt", Ut)
writedlm("Delta_MC_4th_10.txt", Δt)
