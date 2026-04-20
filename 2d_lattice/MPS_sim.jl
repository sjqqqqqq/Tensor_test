using ITensors, ITensorMPS
using JLD2
using Plots

# ── Soft-core two-species boson site type ─────────────────────────────────────
# Local basis: |nₐ, n_b⟩  with  nₐ, n_b ∈ 0..NMAX
# Index ordering: i = nₐ*(NMAX+1) + n_b + 1  (1-based)
const NMAX = 3   # max occupancy per species per site (supports N=1..3 pairs)

ITensors.space(::SiteType"SoftBoson") = (NMAX+1)^2

# Helper: basis index for occupation (na, nb) at one site
occ_idx(na, nb) = na*(NMAX+1) + nb + 1

function ITensors.op(::OpName"Nup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(na); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Ndn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(nb); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Nupdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(na*nb); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdagup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX-1, nb in 0:NMAX
        i = na*d+nb+1; ip = (na+1)*d+nb+1; mat[ip,i] = sqrt(Float64(na+1))
    end; return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 1:NMAX, nb in 0:NMAX
        i = na*d+nb+1; im_ = (na-1)*d+nb+1; mat[im_,i] = sqrt(Float64(na))
    end; return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdagdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX-1
        i = na*d+nb+1; ip = na*d+nb+2; mat[ip,i] = sqrt(Float64(nb+1))
    end; return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 1:NMAX
        i = na*d+nb+1; im_ = na*d+nb; mat[im_,i] = sqrt(Float64(nb))
    end; return ITensor(mat, s', dag(s))
end

const BONDS_A = [(1,2), (3,4)]
const BONDS_B = [(1,3), (2,4)]

# Per-site onsite coefficient for a-type (j is 1-indexed ITensor site)
va_coeff(j, Va1,Va2,Va3) = j==1 ? -(Va1+Va2+Va3) :
                            j==2 ?   Va1 :
                            j==3 ?   Va2 : Va3
vb_coeff(j, Vb1,Vb2,Vb3) = j==1 ? -(Vb1+Vb2+Vb3) :
                             j==2 ?   Vb1 :
                             j==3 ?   Vb2 : Vb3

function make_onsite_gates(s, Va1,Va2,Va3, Vb1,Vb2,Vb3, U, τ)
    [exp(-im*τ*(va_coeff(j,Va1,Va2,Va3)*op("Nup",  s[j]) +
                vb_coeff(j,Vb1,Vb2,Vb3)*op("Ndn",  s[j]) +
                U                       *op("Nupdn",s[j]))) for j in 1:4]
end

make_hop_a(s,j,k,Ja,τ) = exp(-im*τ*Ja*(op("Cdagup",s[j])*op("Cup",   s[k]) +
                                        op("Cup",   s[j])*op("Cdagup",s[k])))
make_hop_b(s,j,k,Jb,τ) = exp(-im*τ*Jb*(op("Cdagdn",s[j])*op("Cdn",   s[k]) +
                                        op("Cdn",   s[j])*op("Cdagdn",s[k])))

function trotter_step(s, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, step_dt;
                      cutoff, maxdim)
    d1h   = make_onsite_gates(s, Va1,Va2,Va3,Vb1,Vb2,Vb3,U, step_dt/2)
    jaAh  = [make_hop_a(s,j,k,Ja,step_dt/2) for (j,k) in BONDS_A]
    jaBf  = [make_hop_a(s,j,k,Ja,step_dt)   for (j,k) in BONDS_B]
    jbAh  = [make_hop_b(s,j,k,Jb,step_dt/2) for (j,k) in BONDS_A]
    jbBf  = [make_hop_b(s,j,k,Jb,step_dt)   for (j,k) in BONDS_B]
    psi = apply(vcat(d1h, jaAh, jaBf, jaAh, jbAh, jbBf, jbAh, d1h), psi;
                cutoff, maxdim)
    normalize!(psi)
    return psi
end

# Build a product-state MPS from a list of (na, nb) per site.
function product_mps(s, occs)
    L = length(s)
    @assert length(occs) == L
    # Link indices of dimension 1 between each pair of sites
    links = [Index(1, "Link,l=$j") for j in 1:L-1]
    tensors = Vector{ITensor}(undef, L)
    for j in 1:L
        na, nb = occs[j]
        inds = if j == 1
            (s[j], links[1])
        elseif j == L
            (links[L-1], s[j])
        else
            (links[j-1], s[j], links[j])
        end
        T = ITensor(ComplexF64, inds...)
        # Set the single nonzero entry (all link indices take value 1).
        if j == 1
            T[s[j] => occ_idx(na,nb), links[1] => 1] = 1.0
        elseif j == L
            T[links[L-1] => 1, s[j] => occ_idx(na,nb)] = 1.0
        else
            T[links[j-1] => 1, s[j] => occ_idx(na,nb), links[j] => 1] = 1.0
        end
        tensors[j] = T
    end
    return MPS(tensors)
end

function build_states(s, N::Int; cutoff, maxdim)
    init_occ = [(N, 0), (0, N), (0, 0), (0, 0)]
    tL_occ   = [(N, 0), (0, N), (0, 0), (0, 0)]
    tR_occ   = [(0, 0), (0, 0), (N, 0), (0, N)]
    psi0   = product_mps(s, init_occ)
    psi_tL = product_mps(s, tL_occ)
    psi_tR = product_mps(s, tR_occ)
    psi_target = normalize!(add(psi_tL, psi_tR; cutoff, maxdim))
    return psi0, psi_target
end

function simulate_case(N::Int; cutoff=1e-10, maxdim=64)
    println("="^70)
    println("MPS simulation — N = $N pair(s)")
    println("="^70)

    pulse_file = "data/GRAPE_2d_$(N).jld2"
    println("Loading GRAPE pulse from: $pulse_file")
    pulse   = load(pulse_file)
    n_pulse = Int(pulse["n"])
    nsteps  = n_pulse - 1
    dt      = pulse["dt"]
    T       = pulse["T"]
    Va1_p   = pulse["Va1"][1:nsteps]; Va2_p = pulse["Va2"][1:nsteps]; Va3_p = pulse["Va3"][1:nsteps]
    Vb1_p   = pulse["Vb1"][1:nsteps]; Vb2_p = pulse["Vb2"][1:nsteps]; Vb3_p = pulse["Vb3"][1:nsteps]
    U_p     = pulse["U"][1:nsteps];   Ja_p  = pulse["Ja"][1:nsteps];  Jb_p  = pulse["Jb"][1:nsteps]
    controls = hcat(Va1_p, Va2_p, Va3_p, Vb1_p, Vb2_p, Vb3_p, U_p, Ja_p, Jb_p)
    grape_fidelity = pulse["fidelity"]
    println("  n_pulse=$n_pulse, T=$(round(T,digits=4)), dt=$(round(dt,digits=6))")
    println("  GRAPE (exact) fidelity: $(round(grape_fidelity, digits=8))")

    s = siteinds("SoftBoson", 4)
    psi0, psi_target = build_states(s, N; cutoff, maxdim)

    println("Initial state norm    : ", norm(psi0))
    println("Initial fidelity F(0) : ", abs2(inner(psi_target, psi0)))

    # JIT warmup
    psi_warm = copy(psi0)
    for n in 1:min(3, nsteps)
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[n,:]
        psi_warm = trotter_step(s, psi_warm, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt;
                                cutoff, maxdim)
    end

    fidelities = Vector{Float64}(undef, nsteps+1)
    bond_dims  = Vector{Int}(undef,     nsteps+1)
    fidelities[1] = abs2(inner(psi_target, psi0))
    bond_dims[1]  = maxlinkdim(psi0)

    psi = copy(psi0)
    t_evolve = @elapsed for n in 1:nsteps
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = controls[n,:]
        psi = trotter_step(s, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt;
                           cutoff, maxdim)
        fidelities[n+1] = abs2(inner(psi_target, psi))
        bond_dims[n+1]  = maxlinkdim(psi)
    end

    println("─── Time evolution complete ───")
    println("Wall time : $(round(t_evolve, digits=2)) s  ($(round(t_evolve/nsteps*1000, digits=3)) ms/step)")
    println("Final F(T): $(fidelities[end])")
    println("Max χ     : $(maximum(bond_dims))")
    println("GRAPE F   : $(round(grape_fidelity, digits=8))")
    println("|ΔF|      : $(round(abs(fidelities[end] - grape_fidelity), sigdigits=3))")

    # Plots
    t_grid = (0:nsteps) .* dt
    ctrl_names = ["Va1","Va2","Va3","Vb1","Vb2","Vb3","U","Ja","Jb"]
    ctrl_plots = [plot(t_grid[1:nsteps], controls[:,k]; title=ctrl_names[k],
                       xlabel="t", ylabel=ctrl_names[k], legend=false, lw=1.2)
                  for k in 1:9]
    p_ctrl = plot(ctrl_plots..., layout=(3,3), size=(900,700),
                  plot_title="MPS sim controls (N=$N)")
    p_fid = plot(t_grid, fidelities;
                 xlabel="t", ylabel="F(t)",
                 title="MPS Fidelity (N=$N, F(T)=$(round(fidelities[end], digits=4)))",
                 legend=false, lw=1.5, ylim=(0, 1.05), color=:crimson)
    fig = plot(p_ctrl, p_fid; layout=grid(2,1, heights=[0.65, 0.35]), size=(900,900))
    out = "figures/MPS_sim_N$(N).png"
    savefig(fig, out)
    println("  Saved $out")
    return (fidelities=fidelities, bond_dims=bond_dims, grape_fidelity=grape_fidelity)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for N in 1:3
        simulate_case(N)
    end
end
