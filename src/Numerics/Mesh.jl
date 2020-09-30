include("Elements.jl")
include("Topology.jl")
import PyPlot
# structure mesh

#=
Nz
Nz-1
Nz-2
.
.
2     e(Nx+1)   e(Nx+2)   e(Nx+3)        e(2Nx)
1     e1        e2        e3       ...   eNx
=
j    
i= 1        2         3        ...    Nx

=#

# DiscontinuousSpectralElementGrid
mutable struct Mesh 
    dim::Int64
    
    Nx::Int64
    Nz::Int64
    Nl::Int64
    Nq::Int64
    
    # topology_type
    topology_type::String
    # the size of the computational domain Lx, Ly for LES configuration and r, R for GCM configuration
    topology_size::Array{Float64, 1}
    # warped topology = (dim=2, (Nl-1)×Nx-1, Nz+1)
    topology::Array{Float64, 3}
    # ghost point locations = (dim=2, Nx)
    top_ghost_cell::Array{Float64, 2}
    # vertical mesh size = (Nl, Nx, Nz)
    Δzc::Array{Float64, 3}
    
   

    # horizontal and vertical min distance for each nodal point (dim=2, Nl×Nx, Nz)
    Δs_min::Array{Float64, 3}
    
    
    #### Gauss-Legendre-Lobatto basis related quantities
    
    # 1-D Gauss-Legendre-Lobatto reference positions (horizontal)
    ξl::Array{Float64, 1}
    # 1-D Gauss-Legendre-Lobatto weights (horizontal)
    ωl::Array{Float64, 1}
    # Nl × Nl = [Dϕ_1 ; Dϕ_2 ; ... ; Dϕ_{N_l}], (horizontal) derivatives at Gauss-Legendre-Lobatto reference positions
    Dl_l::Array{Float64, 2}
    
    
    # 1-D Gauss-Legendre reference positions
    ξq::Array{Float64, 1}
    # 1-D Gauss-Legendre weights
    ωq::Array{Float64, 1}
    
    # Nq × Nl = [ϕ_1 ; ϕ_2 ; ... ; ϕ_{N_l}],  derivatives at Gauss-Legendre reference positions
    ϕl_q::Array{Float64, 2}
    
    # Nq × Nl = [Dϕ_1 ; Dϕ_2 ; ... ; Dϕ_{N_l}],  derivatives at Gauss-Legendre reference positions  
    Dl_q::Array{Float64, 2}
    
    
    #### Geometry related quantities
    
    # Volume integration quantities on each Gauss-Legendre-Lobatto point
    # vol_l_geo = (_nvlgeo=3, Nl, nelem)
    # x1, x2, M
    vol_l_geo::Array{Float64, 3}
    
    # Volume integration quantities on each volume integration point
    # vol_q_geo = (_nvqgeo=5, Nq, nelem)
    # M ∂ξ/∂x ∂ξ/∂z ∂η/∂x ∂η/∂z, x1, x2
    vol_q_geo::Array{Float64, 3}
    
    # sgeo_h = (_nsgeoh=3, Nq=1, nface=2, nelem) for horizontal flux (left and right faces)
    # n1, n2, sM, x1, x2
    sgeo_h::Array{Float64, 4}
    
    # sgeo = (_nsgeo=3, Nl, nface=2, nelem) for vertical flux (bottom and top faces)
    # n1, n2, sM, x1, x2
    sgeo_v::Array{Float64, 4}
    
    
end


function Mesh(Nx::Int64, Nz::Int64, Nl::Int64, Nq::Int64, topology_type::String, topology_size::Array{Float64, 1}, topology::Array{Float64, 3}, top_ghost_cell::Array{Float64, 2})
    dim = size(topology, 1)
    
    @assert(dim == 2)
    @assert(Nx == Int64((size(topology, 2)-1)/(Nl-1)) && Nz == size(topology, 3)-1)
    
    (ξl, ωl) = lglpoints(Nl - 1)
    (ξq, ωq) = lgpoints(Nq - 1)
    
    
    
    wbl  = baryweights(ξl)
    Dl_l = spectralderivative(ξl, ξl, wbl)
    ϕl_q = spectralinterpolate(ξl, ξq, wbl)
    Dl_q = spectralderivative(ξl, ξq, wbl)
    
    
    

    (vol_l_geo, vol_q_geo, sgeo_h, sgeo_v) = compute_geometry(topology, ωl, ωq, Dl_l, ϕl_q, Dl_q)

    Δzc = compute_vertical_mesh_size(Nx, Nz, sgeo_v)

    Δs_min = compute_min_nodal_dist(Nx, Nz, vol_l_geo)
    
    
    Mesh(dim, Nx, Nz, Nl, Nq, topology_type, topology_size, topology, top_ghost_cell, Δzc, Δs_min, 
         ξl, ωl, Dl_l, ξq, ωq, ϕl_q, Dl_q, vol_l_geo, vol_q_geo, sgeo_h, sgeo_v)
end

function compute_vertical_mesh_size(Nx::Int64, Nz::Int64, sgeo_v::Array{Float64, 4})
    
    # sgeo = (_nsgeo=3, Nl, nface=2, nelem) for vertical flux (bottom and top faces)
    # n1, n2, sM, x1, x2
    sgeo_v::Array{Float64, 4}
    
    _nsgeo, Nl, nface, nelem = size(sgeo_v)

    Δzc = zeros(Float64, Nl, Nx, Nz)
    
    # horizontal direction
    for ix = 1:Nx
        for iz = 1:Nz
            e  = ix + (iz-1)*Nx

            for il = 1:Nl
    
                x⁻, z⁻ = sgeo_v[4:5, il, 1, e]
                x⁺, z⁺ = sgeo_v[4:5, il, 2, e]   
                Δz = sqrt((x⁺ - x⁻)^2 + (z⁺ - z⁻)^2)
                Δzc[il, ix, iz] = Δz
                                
            end
        end
    end

    return Δzc

end


function compute_min_nodal_dist(Nx::Int64, Nz::Int64, vol_l_geo::Array{Float64, 3})
    
    _, Nl, nelem = size(vol_l_geo)

    Δs_min = zeros(2, Nl, nelem)
    fill!(Δs_min, Inf64)
    
    # horizontal direction
    for ix = 1:Nx
        for iz = 1:Nz
            
            e  = ix + (iz-1)*Nx
            # left element
            e⁻ = mod1(ix - 1, Nx) + (iz-1)*Nx
            # right element
            e⁺ = mod1(ix + 1, Nx) + (iz-1)*Nx
            for il = 1:Nl
                
                if il == 1
                    x , z  = vol_l_geo[1:2, il, e]
                    x⁺, z⁺ = vol_l_geo[1:2, il+1, e]   
                    Δs⁺ = sqrt((x⁺ - x)^2 + (z⁺ - z)^2)

                    x , z  = vol_l_geo[1:2, Nl, e⁻]
                    x⁻, z⁻ = vol_l_geo[1:2, Nl-1, e⁻]   
                    Δs⁻ = sqrt((x⁻ - x)^2 + (z⁻ - z)^2)
                    
                    Δs_min[1, il, e] = min(Δs⁻, Δs⁺)
                elseif il == Nl
                    x , z  = vol_l_geo[1:2, il, e]
                    x⁻, z⁻ = vol_l_geo[1:2, il-1, e]
                    Δs⁻ = sqrt((x⁻ - x)^2 + (z⁻ - z)^2)

                    x , z  = vol_l_geo[1:2, 1, e⁺]
                    x⁺, z⁺ = vol_l_geo[1:2, 2, e⁺]   
                    Δs⁺ = sqrt((x⁺ - x)^2 + (z⁺ - z)^2)
                    
                    Δs_min[1, il, e] = min(Δs⁻, Δs⁺)
                else
                    x , z  = vol_l_geo[1:2, il, e]
                    x⁺, z⁺ = vol_l_geo[1:2, il+1, e]   
                    Δs⁺ = sqrt((x⁺ - x)^2 + (z⁺ - z)^2)
                    x⁻, z⁻ = vol_l_geo[1:2, il-1, e]
                    Δs⁻ = sqrt((x⁻ - x)^2 + (z⁻ - z)^2)

                    Δs_min[1, il, e] = min(Δs⁻, Δs⁺)
                end

            end
        end
    end

    # vertical direction
    if Nz == 1
        return Δs_min
    end

    for iz = 1:Nz
        for ix = 1:Nx
            e = ix + (iz-1)*Nx
            # bottom element
            e⁻ = ix + mod(iz - 2, Nz)*Nx
            # top element
            e⁺ = ix + mod(iz, Nz)*Nx
            for il = 1:Nl
                
                if iz == 1 
                    x , z  = vol_l_geo[1:2, il, e]
                    x⁺, z⁺ = vol_l_geo[1:2, il, e⁺]   

                    Δs⁺ = sqrt((x⁺ - x)^2 + (z⁺ - z)^2)
                    Δs_min[2, il, e] = Δs⁺

                elseif iz == Nz
                    x , z  = vol_l_geo[1:2, il, e]
                    x⁻, z⁻ = vol_l_geo[1:2, il, e⁻]   
                    Δs⁻ = sqrt((x⁻ - x)^2 + (z⁻ - z)^2)
                    Δs_min[2, il, e] = Δs⁻

                else
                    x , z  = vol_l_geo[1:2, il, e]
                    x⁺, z⁺ = vol_l_geo[1:2, il, e⁺]   
                    Δs⁺ = sqrt((x⁺ - x)^2 + (z⁺ - z)^2)
                    x⁻, z⁻ = vol_l_geo[1:2, il, e⁻]
                    Δs⁻ = sqrt((x⁻ - x)^2 + (z⁻ - z)^2)

                    Δs_min[2, il, e] = min(Δs⁻, Δs⁺)
                end
            end
        end
    end

    return Δs_min

end


function compute_geometry(topology::Array{Float64, 3}, 
    ωl::Array{Float64, 1}, ωq::Array{Float64, 1}, 
    Dl_l::Array{Float64, 2}, ϕl_q::Array{Float64, 2}, Dl_q::Array{Float64, 2})
    
    Nl, Nq = length(ωl), length(ωq)
    dim, Nx, Nz = 2, Int64((size(topology, 2)-1)/(Nl-1)), size(topology, 3)-1
    
    nelem = Nx * Nz
    _nvlgeo = 3
    vol_l_geo = zeros(_nvlgeo, Nl, nelem)
    _nvqgeo = 7
    vol_q_geo = zeros(_nvqgeo, Nq, nelem)
    _nsgeo = 5
    nface = 2
    sgeo_h = zeros(_nsgeo, 1,  nface, nelem)
    sgeo_v = zeros(_nsgeo, Nl, nface, nelem)
    
    for ix = 1:Nx
        for iz = 1:Nz
            
            e = ix + (iz-1)*Nx
            
            # xe = (N_l, 2), ze = (N_l, 2)
            xe, ze = topology[1,  (Nl-1)*(ix-1)+1:(Nl-1)*ix+1,  iz:iz+1] , topology[2, (Nl-1)*(ix-1)+1:(Nl-1)*ix+1 , iz:iz+1] 
            
            @assert(size(xe) == (Nl, 2))
            
            
            # vertical Gauss-Legendre-Lobatto (2) points at center and their derivatives
            # (1 - η)/2 and (1 + η)/2
            
            ϕl¹_q  = [0.5;  0.5]
            Dl¹_q  = [-0.5; 0.5]
            ωl¹_q  = 2.0
            
            # Volume quantities at Nl Gauss-Legendre-Lobatto points
            for i = 1: Nl
                x  = xe[i, 1]*ϕl¹_q[1] + xe[i,2]*ϕl¹_q[2]
                z  = ze[i, 1]*ϕl¹_q[1] + ze[i,2]*ϕl¹_q[2]
                
                
                ∂x∂ξ = Dl_l[i, :]' * (xe *  ϕl¹_q)
                ∂x∂η = xe[i, 1]*Dl¹_q[1] + xe[i,2]*Dl¹_q[2]
                
                
                ∂z∂ξ = Dl_l[i, :]' * (ze *  ϕl¹_q)
                ∂z∂η = ze[i, 1]*Dl¹_q[1] + ze[i,2]*Dl¹_q[2]
                
                detJ = abs(∂x∂ξ*∂z∂η - ∂x∂η*∂z∂ξ)
                M = detJ*ωl[i]*ωl¹_q
                
                vol_l_geo[:, i, e] .= x, z, M
            end
            
            
            
            # Volume quantities at Nq Gauss-Legendre points
            for i = 1: Nq
                x  = ϕl_q[i, :]' * (xe * ϕl¹_q)
                z  = ϕl_q[i, :]' * (ze * ϕl¹_q)
                
                ∂x∂ξ = Dl_q[i, :]' * (xe * ϕl¹_q)
                ∂x∂η = ϕl_q[i, :]' * (xe * Dl¹_q)
                
                
                ∂z∂ξ = Dl_q[i, :]' * (ze * ϕl¹_q)
                ∂z∂η = ϕl_q[i, :]' * (ze * Dl¹_q)
                
                detJ = abs(∂x∂ξ*∂z∂η - ∂x∂η*∂z∂ξ)
                M = detJ*ωq[i]*ωl¹_q
                
                ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z = ∂z∂η/detJ, -∂x∂η/detJ, -∂z∂ξ/detJ, ∂x∂ξ/detJ
                
                
                # J    = [∂x∂ξ ∂x∂η; ∂z∂ξ ∂z∂η]
                # Jinv = [∂ξ∂x ∂ξ∂z; ∂η∂x ∂η∂z]
                # @info J * Jinv
                
                vol_q_geo[:, i, e] .= M, ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z, x, z
            end
            
            
            # Edge quantities at Nq Gauss-Legendre-Lobatto points
            # Left edge and right edge
            
            for iface = 1:2
                N = (iface == 1 ? [-1.0 ; 0.0] : [1.0 ; 0.0]) # left
                f_id = (iface == 1 ? 1 : Nl)

                x  = xe[f_id, 1]*ϕl¹_q[1] + xe[f_id,2]*ϕl¹_q[2]
                z  = ze[f_id, 1]*ϕl¹_q[1] + ze[f_id,2]*ϕl¹_q[2]
                
                ∂x∂ξ = Dl_l[f_id, :]' * (xe * ϕl¹_q)
                ∂x∂η = xe[f_id, :]' * Dl¹_q
                
                ∂z∂ξ = Dl_l[f_id, :]' * (ze * ϕl¹_q)
                ∂z∂η = ze[f_id, :]' * Dl¹_q
                
                detJ = abs(∂x∂ξ*∂z∂η - ∂x∂η*∂z∂ξ)
                sM = detJ*ωl¹_q
                
                ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z = ∂z∂η/detJ, -∂x∂η/detJ, -∂z∂ξ/detJ, ∂x∂ξ/detJ
                
                # ∂ξ∂x, ∂η∂x         
                #                *   N
                # ∂ξ∂z, ∂η∂z
                n = [ ∂ξ∂x*N[1] + ∂η∂x*N[2]  ; ∂ξ∂z*N[1] + ∂η∂z*N[2] ]
                
                sgeo_h[:, 1, iface, e] .= n[1], n[2], sM, x, z
                
            end
            
            # bottom edge and top edge
            for iface in [1, 2]
                # the face is    2
                #                1
                ϕl¹_q = (iface == 1 ? [1.0;  0.0] : [0.0;  1.0])
                N     = (iface == 1 ? [0.0; -1.0] : [0.0 ; 1.0]) 
                
                for i = 1:Nl
                    x  = ϕl_q[i, :]' * (xe * ϕl¹_q)
                    z  = ϕl_q[i, :]' * (ze * ϕl¹_q)
                    
                    ∂x∂ξ = Dl_l[i, :]' * (xe * ϕl¹_q)
                    ∂x∂η = xe[i, :]' * Dl¹_q
                    
                    ∂z∂ξ = Dl_l[i, :]' * (ze * ϕl¹_q)
                    ∂z∂η = ze[i, :]' * Dl¹_q
                    
                    detJ = abs(∂x∂ξ*∂z∂η - ∂x∂η*∂z∂ξ)
                    sM = detJ*ωl[i]
                    
                    ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z = ∂z∂η/detJ, -∂x∂η/detJ, -∂z∂ξ/detJ, ∂x∂ξ/detJ
                    
                    # ∂ξ∂x, ∂η∂x         
                    #                *   N
                    # ∂ξ∂z, ∂η∂z
                    n = [ ∂ξ∂x*N[1] + ∂η∂x*N[2]  ; ∂ξ∂z*N[1] + ∂η∂z*N[2] ]
                    
                    # @info "sM compute ", iface, i, ∂x∂ξ, ∂x∂η, ∂z∂ξ, ∂z∂η, n
                    sgeo_v[:, i, iface, e] .= n[1], n[2], sM, x, z
                end
            end 
        end
    end 
    
    return vol_l_geo, vol_q_geo, sgeo_h, sgeo_v
end



"""
state is a (Float64, Nl, nelem)
"""
function visual(mesh::Mesh, state::Array{Float64, 2}, save_file_name::String="None")

    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
    vol_l_geo = mesh.vol_l_geo
    topology = mesh.topology
    # x1, x2, M
    x, z = reshape(vol_l_geo[1,:,:], (Nl * Nx, Nz)) , reshape(vol_l_geo[2,:,:], (Nl * Nx, Nz)) 

    data = reshape(state, (Nl * Nx, Nz))
    
    
    PyPlot.pcolormesh(x, z, data, shading = "gouraud", cmap = "jet")
    PyPlot.colorbar()
    PyPlot.axis("equal")
    if save_file_name != "None"

        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
end







function Mesh_test()
    Np = 4
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)
    Nx, Nz = 8, 4

    # Nx, Nz = 2, 2

    nelem = Nx * Nz
    
    for topology_type in ["AtmoLES", "AtmoGCM"]
        @info "Start Mesh test: ", topology_type
        if topology_type == "AtmoLES"
            
            Lx, Lz = 2.0, 2.0
            topology_size = [Lx, Lz]
            
            topology = topology_les(Nl, Nx, Nz, Lx, Lz)
            S_ref = Lx*Lz
            vedge_l_ref = Lz*[-1.0; 0.0]

        elseif topology_type == "AtmoGCM"
            
            r, R = 2.0, 4.0
            topology_size = [r, R]

            topology = topology_gcm(Nl, Nx, Nz, r, R)
            S_ref = π*(R^2 - r^2)
            vedge_l_ref = (R - r)*[0.0; 1.0]
            
        end

        # vis_topology(Nx, Nz, topology, topology_type*"_top.pdf")
        mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
        
        # check mass 
        M_lumped = @view mesh.vol_l_geo[3, :, :]
        @info "Gauss-Legendre-Lobatto point size is ", size(M_lumped)
        @info "Gauss-Legendre-Lobatto point mass error is ", sum(M_lumped) - S_ref
        
        M_lumped = @view mesh.vol_q_geo[1, :, :]
        @info "Gauss-Legendre point size is ", size(M_lumped)
        @info "Gauss-Legendre point mass error is ", sum(M_lumped) - S_ref
        
        # check weighted edge norm for each Elements
        weighted_e_tot = [0.0; 0.0]
        weighted_e     = [0.0; 0.0]
        sgeo_v, sgeo_h = mesh.sgeo_v, mesh.sgeo_h 
        for e = 1:nelem
            weighted_e .= 0.0
            
            for il = 1:size(sgeo_v, 2)
                for iface = 1:2
                    n1, n2, sM  = sgeo_v[:, il, iface, e]
                    weighted_e += [n1*sM; n2*sM]
                end
            end
            
            for il = 1:size(sgeo_h, 2)
                for iface = 1:2
                    n1, n2, sM  = sgeo_h[:, il, iface, e]
                    weighted_e += [n1*sM; n2*sM]
                end
            end
            
            weighted_e_tot += abs.(weighted_e) 
            
        end
        @info "sum of edge length weighted norm is ", weighted_e_tot


        # check left/right edge length
        vedge_l, vedge_r = [0.0, 0.0], [0.0, 0.0]
        for iz = 1:Nz

            el = 1 + (iz-1)*Nx
            er = Nx + (iz-1)*Nx
            
            
            n1, n2, sM  = sgeo_h[:, 1, 1, el]
            vedge_l += [n1*sM; n2*sM]

            n1, n2, sM  = sgeo_h[:, 1, 2, er]
            vedge_r += [n1*sM; n2*sM]
            
            
        end
       
        @info "left/right edge norm is ", vedge_l_ref - vedge_l, -vedge_l_ref - vedge_r
        
    end
end





# Mesh_test()