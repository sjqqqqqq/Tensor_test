using Combinatorics

#   L (sites) x N (atoms) basis
function make_basis(N,L)
    V=[]
    for subset in combinations(0:N+L-2,L-1)
        subset0=[]
        push!(subset0,subset[1])
        for j in 0:L-3
            push!(subset0,subset[j+2]-subset[j+1]-1)
        end
        push!(subset0, N+L-2-subset[end])
        push!(V,subset0)
    end
    Basis=reverse(V)
    Ind=Dict{String, Int}()
    d=length(V)
    for i in 0:d-1
        x=[Int(t) for t in Basis[i+1]]
        name=string(x)
        Ind[name]=i
    end
    return Basis, Ind
end
