# soft_boson.jl
# Shared "SoftBoson" two-species boson site type and 4-site ring helpers used
# by MPS_sim.jl and MPS_GRAPE.jl.
#
# Local basis per site: |nₐ, n_b⟩ with nₐ, n_b ∈ 0..NMAX
# Index ordering      : i = nₐ*(NMAX+1) + n_b + 1  (1-based)
#
# 4-site ring topology (Gamma4, 0-indexed → 1-indexed ITensor sites):
#   Group A (MPS-adjacent bonds)     : (1,2), (3,4)
#   Group B (MPS-non-adjacent bonds) : (1,3), (2,4)

using ITensors, ITensorMPS

# ── Local Hilbert space ──────────────────────────────────────────────────────
const NMAX = 3   # max occupancy per species per site (supports N = 1..3 pairs)

ITensors.space(::SiteType"SoftBoson") = (NMAX+1)^2

# ── Named states (convenience for N = 1 case: nₐ, n_b ∈ {0,1}) ────────────────
ITensors.state(::StateName"Emp",  ::SiteType"SoftBoson", s::Index) =
    (T = ITensor(s); T[s=>1]       = 1.0; T)
ITensors.state(::StateName"Dn",   ::SiteType"SoftBoson", s::Index) =
    (T = ITensor(s); T[s=>2]       = 1.0; T)
ITensors.state(::StateName"Up",   ::SiteType"SoftBoson", s::Index) =
    (T = ITensor(s); T[s=>NMAX+2]  = 1.0; T)
ITensors.state(::StateName"UpDn", ::SiteType"SoftBoson", s::Index) =
    (T = ITensor(s); T[s=>NMAX+3]  = 1.0; T)

# ── Single-site operators ────────────────────────────────────────────────────
function ITensors.op(::OpName"Nup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(na); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Ndn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(nb); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Nupdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(na*nb); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdagup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 0:NMAX-1, nb in 0:NMAX
        i = na*d+nb+1; ip = (na+1)*d+nb+1; mat[ip,i] = sqrt(Float64(na+1))
    end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 1:NMAX, nb in 0:NMAX
        i = na*d+nb+1; im_ = (na-1)*d+nb+1; mat[im_,i] = sqrt(Float64(na))
    end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdagdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 0:NMAX, nb in 0:NMAX-1
        i = na*d+nb+1; ip = na*d+nb+2; mat[ip,i] = sqrt(Float64(nb+1))
    end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; mat = zeros(d^2, d^2)
    for na in 0:NMAX, nb in 1:NMAX
        i = na*d+nb+1; im_ = na*d+nb; mat[im_,i] = sqrt(Float64(nb))
    end
    return ITensor(mat, s', dag(s))
end

# ── Product-state MPS from occupations (na, nb) per site ─────────────────────
occ_idx(na::Int, nb::Int) = na*(NMAX+1) + nb + 1

function product_mps(s, occs)
    L = length(s); @assert length(occs) == L
    links = [Index(1, "Link,l=$j") for j in 1:L-1]
    tensors = Vector{ITensor}(undef, L)
    for j in 1:L
        na, nb = occs[j]
        inds = j == 1 ? (s[j], links[1]) :
               j == L ? (links[L-1], s[j]) :
                        (links[j-1], s[j], links[j])
        T = ITensor(ComplexF64, inds...)
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

# ── 4-site ring topology ─────────────────────────────────────────────────────
const BONDS_A   = [(1,2), (3,4)]
const BONDS_B   = [(1,3), (2,4)]
const ALL_BONDS = vcat(BONDS_A, BONDS_B)

# ── On-site potential coefficients ───────────────────────────────────────────
# H_onsite = Σ_k Vak(n^a_k − n^a_0) + Σ_k Vbk(n^b_k − n^b_0) + U·Σ_j n^a_j n^b_j
va_coeff(j, Va1,Va2,Va3) = j==1 ? -(Va1+Va2+Va3) :
                           j==2 ?   Va1 :
                           j==3 ?   Va2 : Va3
vb_coeff(j, Vb1,Vb2,Vb3) = j==1 ? -(Vb1+Vb2+Vb3) :
                           j==2 ?   Vb1 :
                           j==3 ?   Vb2 : Vb3

# ── Gate factories ───────────────────────────────────────────────────────────
# sign=+1 → forward gate exp(-iτH); sign=-1 → adjoint gate exp(+iτH)

function make_onsite_gates(s, Va1,Va2,Va3, Vb1,Vb2,Vb3, U, τ; sign::Int=1)
    [exp(sign*(-im)*τ*(va_coeff(j,Va1,Va2,Va3)*op("Nup",  s[j]) +
                       vb_coeff(j,Vb1,Vb2,Vb3)*op("Ndn",  s[j]) +
                       U                        *op("Nupdn",s[j]))) for j in 1:4]
end

make_hop_a(s, j, k, Ja, τ; sign::Int=1) =
    exp(sign*(-im)*τ*Ja*(op("Cdagup",s[j])*op("Cup",   s[k]) +
                         op("Cup",   s[j])*op("Cdagup",s[k])))
make_hop_b(s, j, k, Jb, τ; sign::Int=1) =
    exp(sign*(-im)*τ*Jb*(op("Cdagdn",s[j])*op("Cdn",   s[k]) +
                         op("Cdn",   s[j])*op("Cdagdn",s[k])))

# 2nd-order sub-Trotter for one species: A/2 → B → A/2
hop_gates_a(s, Ja, τ; sign::Int=1) =
    vcat([make_hop_a(s,j,k,Ja,τ/2; sign) for (j,k) in BONDS_A],
         [make_hop_a(s,j,k,Ja,τ  ; sign) for (j,k) in BONDS_B],
         [make_hop_a(s,j,k,Ja,τ/2; sign) for (j,k) in BONDS_A])
hop_gates_b(s, Jb, τ; sign::Int=1) =
    vcat([make_hop_b(s,j,k,Jb,τ/2; sign) for (j,k) in BONDS_A],
         [make_hop_b(s,j,k,Jb,τ  ; sign) for (j,k) in BONDS_B],
         [make_hop_b(s,j,k,Jb,τ/2; sign) for (j,k) in BONDS_A])

# ── SPDC-like default states for N pairs ─────────────────────────────────────
# Initial: all N a-bosons on site 0, all N b-bosons on site 1
# Target : (|all-a@0, all-b@1⟩ + |all-a@2, all-b@3⟩)/√2
function default_states(s, N::Int; cutoff::Float64=1e-10, maxdim::Int=64)
    init_occ = [(N,0), (0,N), (0,0), (0,0)]
    tR_occ   = [(0,0), (0,0), (N,0), (0,N)]
    psi0       = product_mps(s, init_occ)
    psi_target = normalize!(add(product_mps(s, init_occ),
                                product_mps(s, tR_occ); cutoff, maxdim))
    return psi0, psi_target
end
