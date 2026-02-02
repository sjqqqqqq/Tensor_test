using Pkg
# Pkg.activate(".")
Pkg.activate("Tensor_test")

using ITensors, ITensorMPS
using LinearAlgebra, Random, Printf
using Optim

# === Parameters ===
const N, K, T = 20, 3, 10.0          # particles, sites, total time
const NSTEPS = 101                   # time steps
const DT = T / (NSTEPS - 1)
const CUTOFF = 1e-8

# === Setup ===
const sites = siteinds("Boson", K; dim=N+1, conserve_qns=true)
const CENTER = (K + 1) / 2

# Precompute operators
const OP_N = [op("N", sites[j]) for j in 1:K]
const OP_NNmN = [op("N * N", sites[j]) - op("N", sites[j]) for j in 1:K]
const OP_H2 = [op("Adag", sites[j]) * op("A", sites[j+1]) +
               op("A", sites[j]) * op("Adag", sites[j+1]) for j in 1:K-1]

# === States ===
psi0 = MPS(sites, vcat(["$N"], fill("0", K-1)))
psi_target = let
    s1, s2 = vcat(["$N"], fill("0", K-1)), vcat(fill("0", K-1), ["$N"])
    psi = add(MPS(sites, s1), MPS(sites, s2); cutoff=CUTOFF)
    normalize!(psi ./= sqrt(2))
end

# === Gates ===
H1(j, U, Δ) = U * OP_NNmN[j] + Δ * (j - CENTER) * OP_N[j]
make_H1_gates(U, Δ, τ) = [exp(-im * τ * H1(j, U, Δ)) for j in 1:K]
make_H2_gates(J, τ) = [exp(-im * τ * J * OP_H2[j]) for j in 1:K-1]

# === Propagation ===
function forward(psi, J, U, Δ; store=false)
    states = store ? Vector{MPS}(undef, 3(NSTEPS-1)) : nothing
    for n in 1:NSTEPS-1
        g1, g2 = make_H1_gates(U[n], Δ[n], DT/2), make_H2_gates(J[n], DT)
        psi = apply(g1, psi; cutoff=CUTOFF); normalize!(psi)
        store && (states[3n-2] = copy(psi))
        psi = apply(g2, psi; cutoff=CUTOFF); normalize!(psi)
        store && (states[3n-1] = copy(psi))
        psi = apply(g1, psi; cutoff=CUTOFF); normalize!(psi)
        store && (states[3n] = copy(psi))
    end
    store ? (psi, states) : psi
end

function backward(chi, J, U, Δ)
    states = Vector{MPS}(undef, 2(NSTEPS-1))
    for n in NSTEPS-1:-1:1
        g1 = [exp(+im * DT/2 * H1(j, U[n], Δ[n])) for j in K:-1:1]
        g2 = [exp(+im * DT * J[n] * OP_H2[j]) for j in K-1:-1:1]
        chi = apply(g1, chi; cutoff=CUTOFF); normalize!(chi)
        states[2n] = copy(chi)  # before H2
        chi = apply(g2, chi; cutoff=CUTOFF); normalize!(chi)
        states[2n-1] = copy(chi)  # before first H1
        chi = apply(g1, chi; cutoff=CUTOFF); normalize!(chi)
    end
    states
end

# === Expectation values ===
function expect(chi, psi, Op, j; twosite=false)
    chi, psi = orthogonalize(chi, j), orthogonalize(psi, j)
    L = j == 1 ? ITensor(1.0) : reduce((a,i) -> a * dag(chi[i]) * psi[i], 1:j-1; init=ITensor(1.0))
    R = (twosite ? j+1 : j) == K ? ITensor(1.0) :
        reduce((a,i) -> a * dag(chi[i]) * psi[i], K:-1:(twosite ? j+2 : j+1); init=ITensor(1.0))
    if twosite
        scalar(L * dag(chi[j]) * dag(chi[j+1]) * noprime(Op * psi[j] * psi[j+1]) * R)
    else
        scalar(L * dag(chi[j]) * noprime(Op * psi[j]) * R)
    end
end

# === Analytical gradients ===
function gradients(J, U, Δ)
    psi_final, fwd = forward(copy(psi0), J, U, Δ; store=true)
    bwd = backward(copy(psi_target), J, U, Δ)

    overlap = inner(psi_target, psi_final)
    gJ, gU, gΔ = zeros(NSTEPS), zeros(NSTEPS), zeros(NSTEPS)

    for n in 1:NSTEPS-1
        # J gradient (H2 layer)
        for j in 1:K-1
            gJ[n] += 2real(conj(overlap) * (-im*DT) * expect(bwd[2n], fwd[3n-1], OP_H2[j], j; twosite=true))
        end
        # U, Δ gradients (both H1 layers)
        for (chi, psi) in [(bwd[2n-1], fwd[3n-2]), (bwd[2n], fwd[3n-1])]
            for j in 1:K
                gU[n] += 2real(conj(overlap) * (-im*DT/2) * expect(chi, psi, OP_NNmN[j], j))
                gΔ[n] += 2real(conj(overlap) * (-im*DT/2) * expect(chi, psi, (j-CENTER)*OP_N[j], j))
            end
        end
    end
    gJ, gU, gΔ, abs2(overlap)
end

# === Pack/unpack parameters for Optim.jl ===
function pack(J, U, Δ)
    vcat(J, U, Δ)
end

function unpack(x)
    J = x[1:NSTEPS]
    U = x[NSTEPS+1:2NSTEPS]
    Δ = x[2NSTEPS+1:3NSTEPS]
    J, U, Δ
end

# === Cost function for Optim.jl (minimization) ===
function loss(x)
    J, U, Δ = unpack(x)
    J = max.(J, 0.01)  # Enforce J > 0
    psi_final = forward(copy(psi0), J, U, Δ)
    fidelity = abs2(inner(psi_target, psi_final))
    return 1.0 - fidelity  # Minimize infidelity
end

function loss_grad!(G, x)
    J, U, Δ = unpack(x)
    J = max.(J, 0.01)
    gJ, gU, gΔ, _ = gradients(J, U, Δ)
    # Negate gradient (maximization -> minimization)
    G[1:NSTEPS] .= -gJ
    G[NSTEPS+1:2NSTEPS] .= -gU
    G[2NSTEPS+1:3NSTEPS] .= -gΔ
end

# === Optimization with L-BFGS ===
function optimize(; maxiter=150, tol=1e-5, warmstart=nothing)
    Random.seed!(42)
    t = range(0, T, length=NSTEPS)

    # Initialize or warm-start
    if warmstart !== nothing
        J0, U0, Δ0 = warmstart
        println("Warm-starting from previous solution")
    else
        J0 = 1.0 .+ 0.3sin.(π*t/T)
        U0 = 0.1 .* ones(NSTEPS)
        Δ0 = 0.3sin.(2π*t/T)
    end

    x0 = pack(J0, U0, Δ0)
    best_fid = Ref(0.0)
    iter_count = Ref(0)

    function callback(state)
        iter_count[] += 1
        fid = 1.0 - state.f_x
        if fid > best_fid[]
            best_fid[] = fid
        end
        if iter_count[] == 1 || iter_count[] % 10 == 0
            @printf("Iter %3d: F = %.6f (best = %.6f)\n", iter_count[], fid, best_fid[])
        end
        # Early stopping if very high fidelity
        fid > 0.999 && return true
        return false
    end

    result = Optim.optimize(
        loss, loss_grad!, x0,
        LBFGS(m=30),
        Optim.Options(
            iterations=maxiter,
            g_tol=1e-10,
            f_reltol=tol,
            show_trace=false,
            callback=callback
        )
    )

    J, U, Δ = unpack(Optim.minimizer(result))
    J .= max.(J, 0.01)
    fidelity = 1.0 - Optim.minimum(result)

    println("\nL-BFGS converged: ", Optim.converged(result))
    println("Iterations: ", Optim.iterations(result))

    J, U, Δ, fidelity
end

# === Main ===

println("GRAPE MPS Demo: $N bosons, $K sites, T=$T")
println("Target: NOON state (|$N,0,0⟩ + |0,0,$N⟩)/√2\n")

@time J, U, Δ, F = optimize()

println("\nFinal fidelity: $F")
psi_final = forward(copy(psi0), J, U, Δ)
println("P(|$N,0,0⟩) = ", abs2(inner(MPS(sites, vcat(["$N"], fill("0", K-1))), psi_final)))
println("P(|0,0,$N⟩) = ", abs2(inner(MPS(sites, vcat(fill("0", K-1), ["$N"])), psi_final)))

# Save results
open("J_opt.txt", "w") do f; foreach(v -> println(f, v), J); end
open("U_opt.txt", "w") do f; foreach(v -> println(f, v), U); end
open("Delta_opt.txt", "w") do f; foreach(v -> println(f, v), Δ); end
println("\nControl pulses saved to J_opt.txt, U_opt.txt, Delta_opt.txt")
