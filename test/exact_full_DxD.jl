# exact_full_DxD.jl
# Replay a saved GRAPE pulse using FULL D×D dense matrices.
# Intentionally inefficient: builds the D×D Kronecker hopping matrices
# kron(h_a, I_b), kron(I_a, h_b) and exponentiates each Trotter sub-block
# as a D×D matrix. Outputs the fidelity trajectory F(t) = |⟨ψ_target|ψ(t)⟩|².
#
# Usage:
#   julia exact_full_DxD.jl path/to/GRAPE_2d_Npair_N.jld2 [out.jld2]

using JLD2, LinearAlgebra, Printf

include(joinpath(@__DIR__, "..", "2d_lattice", "exact_GRAPE.jl"))

function build_full_matrices(N::Int)
    basis_a = species_basis(L_SITES, N)
    basis_b = species_basis(L_SITES, N)
    Da = length(basis_a); Db = length(basis_b); D = Da * Db

    h_a = species_hopping(basis_a)
    h_b = species_hopping(basis_b)
    Ia = Matrix{ComplexF64}(I, Da, Da)
    Ib = Matrix{ComplexF64}(I, Db, Db)

    H_Ja_mat = kron(h_a, Ib)   # D × D
    H_Jb_mat = kron(Ia, h_b)   # D × D

    n_a = [species_occupation(basis_a, j) for j in 0:L_SITES-1]
    n_b = [species_occupation(basis_b, j) for j in 0:L_SITES-1]
    dVa = [kron(n_a[k+1] .- n_a[1], ones(Db)) for k in 1:3]
    dVb = [kron(ones(Da), n_b[k+1] .- n_b[1]) for k in 1:3]
    dU  = zeros(Float64, D)
    for j in 1:L_SITES
        dU .+= kron(n_a[j], n_b[j])
    end

    return (Da=Da, Db=Db, D=D, basis_a=basis_a, basis_b=basis_b,
            H_Ja=H_Ja_mat, H_Jb=H_Jb_mat,
            dVa=dVa, dVb=dVb, dU=dU)
end

function trotter_step_full(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
    h = Va1 .* sys.dVa[1] .+ Va2 .* sys.dVa[2] .+ Va3 .* sys.dVa[3] .+
        Vb1 .* sys.dVb[1] .+ Vb2 .* sys.dVb[2] .+ Vb3 .* sys.dVb[3] .+
        U   .* sys.dU
    d_half = exp.(-im * (dt/2) .* h)

    UJa = exp(-im * Ja * dt * Matrix(sys.H_Ja))   # D × D matrix exp
    UJb = exp(-im * Jb * dt * Matrix(sys.H_Jb))   # D × D matrix exp

    psi = d_half .* psi
    psi = UJa * psi
    psi = UJb * psi
    psi = d_half .* psi
    return psi
end

function fidelity_trajectory_full(sys, psi0, psi_target, ctrls, dt)
    nsteps = size(ctrls, 1)
    fid = zeros(Float64, nsteps)
    psi = Vector{ComplexF64}(psi0)
    fid[1] = abs2(dot(psi_target, psi))
    for n in 1:nsteps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n, :]
        psi = trotter_step_full(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
        fid[n+1] = abs2(dot(psi_target, psi))
    end
    return fid
end

function run(infile::String; outfile::Union{String,Nothing}=nothing)
    d = load(infile)
    N = haskey(d, "Npair") ? d["Npair"] : d["N"]
    n = d["n"]; T = d["T"]
    Va1,Va2,Va3 = d["Va1"], d["Va2"], d["Va3"]
    Vb1,Vb2,Vb3 = d["Vb1"], d["Vb2"], d["Vb3"]
    U, Ja, Jb   = d["U"],   d["Ja"],  d["Jb"]
    Fopt = get(d, "fidelity", NaN)
    ctrls = hcat(Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb)
    dt = T / (n - 1)

    sys = build_full_matrices(N)
    psi0 = zeros(ComplexF64, sys.D)
    psi0[(species_index(sys.basis_a, fill(0, N)) - 1) * sys.Db +
          species_index(sys.basis_b, fill(1, N))] = 1.0
    psi_R = zeros(ComplexF64, sys.D)
    psi_R[(species_index(sys.basis_a, fill(2, N)) - 1) * sys.Db +
          species_index(sys.basis_b, fill(3, N))] = 1.0
    psi_target = (psi0 .+ psi_R) ./ sqrt(2.0)

    @printf("Full D×D replay: N=%d, Da=%d, Db=%d, D=%d, n=%d, T=%.4f\n",
            N, sys.Da, sys.Db, sys.D, n, T)
    @printf("Saved F = %s\n", isnan(Fopt) ? "n/a" : string(round(Fopt, digits=6)))

    t0 = time()
    fid = fidelity_trajectory_full(sys, psi0, psi_target, ctrls, dt)
    wall = time() - t0
    @printf("Recomputed F(0) = %.6f, F(T) = %.6f   wall = %.3f s\n",
            fid[1], fid[end], wall)

    t = collect(range(0, T, length=n))
    out = isnothing(outfile) ?
          joinpath(@__DIR__, "data", "exact_full_DxD_N$(N).jld2") :
          outfile
    mkpath(dirname(out))
    jldsave(out; N=N, T=T, n=n, dt=dt, t=t, fid=fid, F_final=fid[end], wall=wall)
    println("Saved $out")
    return fid
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) ≥ 1 || error("usage: julia exact_full_DxD.jl <input.jld2> [output.jld2]")
    run(ARGS[1]; outfile = length(ARGS) ≥ 2 ? ARGS[2] : nothing)
end
