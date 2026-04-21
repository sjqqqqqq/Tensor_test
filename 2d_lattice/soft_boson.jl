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
#
# Performance notes (see CLAUDE.md § "Performance notes" for context):
#   • Onsite gates are built as a diagonal exp in the occupation basis —
#     no call to `exp(::ITensor)` — because H_onsite is diagonal there.
#   • Hopping gates use an eigendecomposition cache: each bond's generator
#     H_{jk} = C†_j C_k + C_j C†_k is Hermitian and control-independent, so
#     we decompose it once (V·Diag(λ)·V†) and rebuild
#         exp(-iτ·J·H) = V·Diag(exp(-iτ·J·λ))·V†
#     on every call. Two matmuls replace a 256×256 Padé matrix exp.
#   • The cache is keyed by siteinds identity (IdDict); hop_generator_tensors
#     exposes the generator ITensors themselves so MPS_GRAPE.jl's hop_inner
#     calls don't rebuild `op()*op()` inside the hot loop.

using ITensors, ITensorMPS
using LinearAlgebra

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

# Onsite generator is diagonal in the occupation basis → build exp directly
# without calling exp(::ITensor). ~1 ms → ~0.05 ms per step total.
function make_onsite_gates(s, Va1,Va2,Va3, Vb1,Vb2,Vb3, U, τ; sign::Int=1)
    d = NMAX+1
    gates = Vector{ITensor}(undef, 4)
    coef = sign*(-im)*τ
    for j in 1:4
        va = va_coeff(j, Va1, Va2, Va3)
        vb = vb_coeff(j, Vb1, Vb2, Vb3)
        diagvec = Vector{ComplexF64}(undef, d*d)
        @inbounds for na in 0:NMAX, nb in 0:NMAX
            i = na*d + nb + 1
            diagvec[i] = exp(coef * (va*na + vb*nb + U*na*nb))
        end
        M = Diagonal(diagvec)
        gates[j] = ITensor(Matrix(M), s[j]', dag(s[j]))
    end
    return gates
end

# Hopping generator eigendecomposition cache.
# Each hopping generator H_{jk} = C†_j C_k + C_j C†_k is control-independent,
# so we eigendecompose once per (species, bond) and rebuild exp(-iτJH) via
# V·Diag(exp(-iτJλ))·V† on every call (≈10× faster than matrix exp of 256×256).
struct _HopEntry
    V::Matrix{ComplexF64}
    λ::Vector{Float64}
    ij::Index
    ik::Index
    d::Int
    H::ITensor   # cached generator tensor H_{jk} for use in ⟨chi|H|psi⟩ calls
end

struct _HopCache
    a_A::Vector{_HopEntry}
    a_B::Vector{_HopEntry}
    b_A::Vector{_HopEntry}
    b_B::Vector{_HopEntry}
end

function _hop_entry(s, j::Int, k::Int, species::Symbol)
    if species === :up
        H = op("Cdagup",s[j])*op("Cup",   s[k]) +
            op("Cup",   s[j])*op("Cdagup",s[k])
    else
        H = op("Cdagdn",s[j])*op("Cdn",   s[k]) +
            op("Cdn",   s[j])*op("Cdagdn",s[k])
    end
    ij, ik = s[j], s[k]
    d = dim(ij)
    M = Array(H, ij', ik', ij, ik)          # shape (d,d,d,d)
    Mflat = Matrix(reshape(M, d*d, d*d))
    Mflat = (Mflat + Mflat') / 2             # Hermitize
    F = eigen(Hermitian(Mflat))
    return _HopEntry(Matrix{ComplexF64}(F.vectors), Float64.(F.values), ij, ik, d, H)
end

# Public helper: cached generator ITensors per species, keyed by bond index
# in `ALL_BONDS` order ([(1,2),(3,4),(1,3),(2,4)]).
function hop_generator_tensors(s, species::Symbol)
    c = _get_hop_cache(s)
    if species === :up
        return vcat([e.H for e in c.a_A], [e.H for e in c.a_B])
    else
        return vcat([e.H for e in c.b_A], [e.H for e in c.b_B])
    end
end

const _HOP_CACHES = IdDict{Any,_HopCache}()

function _get_hop_cache(s)
    get!(_HOP_CACHES, s) do
        _HopCache(
            [_hop_entry(s, j, k, :up) for (j,k) in BONDS_A],
            [_hop_entry(s, j, k, :up) for (j,k) in BONDS_B],
            [_hop_entry(s, j, k, :dn) for (j,k) in BONDS_A],
            [_hop_entry(s, j, k, :dn) for (j,k) in BONDS_B],
        )
    end
end

@inline function _gate_from_eig(ent::_HopEntry, J::Real, τ::Real, sign::Int)
    expD = exp.((sign*(-im)*τ*J) .* ent.λ)
    U = ent.V * Diagonal(expD) * ent.V'
    d = ent.d
    return ITensor(reshape(U, d, d, d, d), ent.ij', ent.ik', ent.ij, ent.ik)
end

# Kept for API compatibility (rarely called on its own)
make_hop_a(s, j, k, Ja, τ; sign::Int=1) =
    _gate_from_eig(_hop_entry(s, j, k, :up), Ja, τ, sign)
make_hop_b(s, j, k, Jb, τ; sign::Int=1) =
    _gate_from_eig(_hop_entry(s, j, k, :dn), Jb, τ, sign)

# 2nd-order sub-Trotter for one species: A/2 → B → A/2
function hop_gates_a(s, Ja, τ; sign::Int=1)
    c = _get_hop_cache(s)
    gs = Vector{ITensor}(undef, 2*length(c.a_A) + length(c.a_B))
    i = 1
    for e in c.a_A; gs[i] = _gate_from_eig(e, Ja, τ/2, sign); i += 1; end
    for e in c.a_B; gs[i] = _gate_from_eig(e, Ja, τ,   sign); i += 1; end
    for e in c.a_A; gs[i] = _gate_from_eig(e, Ja, τ/2, sign); i += 1; end
    return gs
end
function hop_gates_b(s, Jb, τ; sign::Int=1)
    c = _get_hop_cache(s)
    gs = Vector{ITensor}(undef, 2*length(c.b_A) + length(c.b_B))
    i = 1
    for e in c.b_A; gs[i] = _gate_from_eig(e, Jb, τ/2, sign); i += 1; end
    for e in c.b_B; gs[i] = _gate_from_eig(e, Jb, τ,   sign); i += 1; end
    for e in c.b_A; gs[i] = _gate_from_eig(e, Jb, τ/2, sign); i += 1; end
    return gs
end

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
