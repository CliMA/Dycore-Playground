include("FVModel.jl")

@doc """
This is horizontal volume tendency (Cartisian grid)
∫ F(W)∇ϕ_j = ∫ F(W)∇_ξϕ_j J⁻¹ = ∑ F(W_i) ∇_ξ⋅ϕ_j(ξ_i) J⁻¹_i w_i 
""" 


function horizontal_volume_tendency!(
    app::Application,
    mesh::Mesh,
    state_prognostic::Array{Float64, 3},
    state_auxiliary_vol_q::Array{Float64,3},
    tendency::Array{Float64, 3},
    )
    
    # Each element has Nl Gauss-Legendre-Lobatto points(basis), 
    # and Nq quadrature points
    
    Nq , Nl, Nx, Nz = mesh.Nq, mesh.Nl, mesh.Nx, mesh.Nz
    num_state_prognostic = app.num_state_prognostic
    dim, vol_q_geo, ϕl_q, Dl_q = mesh.dim, mesh.vol_q_geo, mesh.ϕl_q, mesh.Dl_q
    
    
    
    
    Threads.@threads for iz = 1:Nz
        
        # reconstructed local state at quadrature points
        local_states_q = zeros(Float64, Nq, num_state_prognostic)
        local_fluxes_q = zeros(Float64, Nq, num_state_prognostic, dim)
        
        local_flux = zeros(Float64, num_state_prognostic)
        
        for ix = 1:Nx
            e = ix + (iz-1)*Nx
            
            local_states_l = @view state_prognostic[:, :, e]
            local_aux_q    = @view state_auxiliary_vol_q[:, :, e]
            
            for s = 1:num_state_prognostic
                local_states_q[:, s] =  ϕl_q * local_states_l[:, s]
            end
            
            for iq = 1:Nq
                local_fluxes_q[iq, :, :] = flux_first_order(app, local_states_q[iq, :], local_aux_q[iq, :])
            end
            
            # Loop Legendre-Gauss-Lobatto points to construct ∇ϕ_j = ∇_ξϕ_j J⁻¹
            for iq = 1:Nq
                M, ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z = vol_q_geo[:, iq, e]
                for il = 1:Nl
                    
                    tendency[il, :, e] .+= local_fluxes_q[iq, :, :] *( M * Dl_q[iq, il] * [∂ξ∂x ; ∂ξ∂z])
                end
            end
            
        end
        
    end
end



function horizontal_volume_tendency!(
    app::Application,
    mesh::Mesh,
    state_prognostic::Array{Float64, 3},
    state_auxiliary_vol_l::Array{Float64,3},
    state_auxiliary_vol_q::Array{Float64,3},
    tendency::Array{Float64, 3},
    )
    
    # Each element has Nl Gauss-Legendre-Lobatto points(basis), 
    # and Nq quadrature points
    
    Nq , Nl, Nx, Nz = mesh.Nq, mesh.Nl, mesh.Nx, mesh.Nz
    num_state_prognostic = app.num_state_prognostic
    dim, vol_q_geo, ϕl_q, Dl_q = mesh.dim, mesh.vol_q_geo, mesh.ϕl_q, mesh.Dl_q
    
    
    
    
    p_state_id, p_aux_id = 4, 4
    
    Threads.@threads for iz = 1:Nz

        # reconstructed local state at quadrature points
    local_states_q = zeros(Float64, Nq, num_state_prognostic)
    local_fluxes_q = zeros(Float64, Nq, num_state_prognostic, dim)
    local_states_prim_q = zeros(Float64, Nq, num_state_prognostic)
    local_states_prim_l = zeros(Float64, Nl, num_state_prognostic)
    
    local_flux = zeros(Float64, num_state_prognostic)

    
        for ix = 1:Nx
            e = ix + (iz-1)*Nx
            
            local_states_l = @view state_prognostic[:, :, e]
            local_aux_l    = @view state_auxiliary_vol_l[:, :, e]
            local_aux_q    = @view state_auxiliary_vol_q[:, :, e]
            
            for il = 1:Nl
                local_states_prim_l[il, :] = prog_to_prim(app, local_states_l[il, :], local_aux_l[il, :])
            end
            
            for s = 1:num_state_prognostic
                if s == p_state_id
                    local_states_prim_q[:, s] =  ϕl_q * (local_states_prim_l[:, s] - local_aux_l[:, p_aux_id]) + local_aux_q[:, p_aux_id]
                else
                    local_states_prim_q[:, s] =  ϕl_q * local_states_prim_l[:, s]
                end
            end
            
            for iq = 1:Nq
                local_states_q[iq, :] = prim_to_prog(app, local_states_prim_q[iq, :], local_aux_q[iq, :])
            end
            
            for iq = 1:Nq
                local_fluxes_q[iq, :, :] = flux_first_order(app, local_states_q[iq, :], local_aux_q[iq, :])
            end
            
            # Loop Legendre-Gauss-Lobatto points to construct ∇ϕ_j = ∇_ξϕ_j J⁻¹
            for iq = 1:Nq
                M, ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z = vol_q_geo[:, iq, e]
                for il = 1:Nl
                    
                    tendency[il, :, e] .+= local_fluxes_q[iq, :, :] *( M * Dl_q[iq, il] * [∂ξ∂x ; ∂ξ∂z])
                end
            end
            
        end
        
    end
end


@doc """
interface_tendency!(balance_law::BalanceLaw, Val(polyorder),
numerical_flux_first_order,
numerical_flux_second_order,
tendency, state_prognostic, state_gradient_flux, state_auxiliary,
vgeo, sgeo, t, vmap⁻, vmap⁺, elemtobndy,
elems)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.
""" interface_tendency!
function horizontal_interface_tendency!(
    app::Application,
    mesh::Mesh,
    state_prognostic::Array{Float64, 3},
    state_auxiliary_surf_h::Array{Float64,4},
    tendency::Array{Float64, 3}
    ) 
    
    Nx, Nz = mesh.Nx, mesh.Nz
    bc_left_type, bc_left_data = app.bc_left_type, app.bc_left_data
    bc_right_type, bc_right_data = app.bc_left_type, app.bc_right_data
    
    sgeo_h = mesh.sgeo_h
    
    Threads.@threads for iz = 1:Nz
        # Compute the flux on the ix-th face
        for ix = 1:Nx+1
            
            # left and right element in the periodic sense
            e⁻ = mod1(ix - 1, Nx) + (iz-1)*Nx
            e⁺ = mod1(ix, Nx) + (iz-1)*Nx
            
            # @show e⁻,  e⁺ 
            
            local_state⁻ = state_prognostic[end, :, e⁻]
            local_state⁺ = state_prognostic[1,   :, e⁺]
            
            local_aux⁻ = state_auxiliary_surf_h[1, :, end, e⁻]
            local_aux⁺ = state_auxiliary_surf_h[1, :, 1, e⁺]
            
            if ix == 1 # left bc
                
                (n1, n2, sM) = sgeo_h[:, 1, 1, e⁺] 
                # the normal points from e⁻ to e⁺
                n1, n2 = -n1, -n2
                
                if bc_left_type == "periodic"
                    local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁺, local_state⁺, local_aux⁺, [n1;n2])
                    
                elseif bc_left_type == "no-slip"
                    
                    local_flux = wall_flux_first_order(app, local_state⁺, local_aux⁺, [n1;n2])
                    
                else
                    error("bc_left_type = ", bc_left_type, " has not implemented")   
                end
                
                tendency[1,   :, e⁺] .+= sM * local_flux
                
            elseif ix == Nx + 1 # right bc
                
                (n1, n2, sM) = sgeo_h[:, 1, end, e⁻] 
                
                if bc_right_type == "periodic"
                    local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁻, [n1;n2])
                    
                elseif bc_right_type == "no-slip"
                    
                    local_flux = wall_flux_first_order(app, local_state⁻, local_aux⁻, [n1;n2])
                    
                else
                    error("bc_right_type = ", bc_right_type, " has not implemented") 
                end
                
                tendency[end, :, e⁻] .-= sM * local_flux
                
            else
                
                
                
                (n1, n2, sM) = sgeo_h[:, 1, end, e⁻] 
                
                
                
                local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁺, [n1;n2])
                
                tendency[1,   :, e⁺] .+= sM * local_flux
                tendency[end, :, e⁻] .-= sM * local_flux
                
                
            end
        end
        
    end
end












#ghost cell, reconstruction for primitive or conservative variables
function source_tendency!(
    app::Application,
    mesh::Mesh,
    state_prognostic::Array{Float64, 3},
    state_auxiliary_vol_l::Array{Float64,3},
    tendency::Array{Float64, 3}
    )   
    
    vol_l_geo = mesh.vol_l_geo
    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
    Threads.@threads for iz = 1:Nz
        for ix = 1:Nx
            e = ix + (iz-1)*Nx
            
            local_states_l = @view state_prognostic[:, :, e]
            local_aux_l = @view state_auxiliary_vol_l[:, :, e]
            
            for il = 1:Nl
                x, z, M = vol_l_geo[:, il, e]
                
                
                tendency[il, :, e] += source(app, local_states_l[il, :], local_aux_l[il, :], x) * M 
            end
        end
        
    end
end





# #ghost cell, reconstruction for primitive or conservative variables
# function geopotential_source_tendency!(
#     app::Application,
#     mesh::Mesh,
#     state_prognostic::Array{Float64, 3},
#     state_auxiliary_vol_l::Array{Float64,3},
#     state_auxiliary_vol_q::Array{Float64,3},    # auxiliary states at volume Gauss-Legendre points
#     state_auxiliary_surf_h::Array{Float64,4},   # auxiliary states at horizontal flux surface
#     state_auxiliary_surf_v::Array{Float64,4}, 
#     tendency::Array{Float64, 3})



#     # Each element has Nl Gauss-Legendre-Lobatto points(basis), 
#     # and Nq quadrature points

#     Nq , Nl, Nx, Nz = mesh.Nq, mesh.Nl, mesh.Nx, mesh.Nz
#     num_state_prognostic = app.num_state_prognostic
#     dim, vol_q_geo, ϕl_q, Dl_q = mesh.dim, mesh.vol_q_geo, mesh.ϕl_q, mesh.Dl_q


#     # reconstructed local state at quadrature points
#     local_states_q = zeros(Float64, Nq, num_state_prognostic)
#     local_fluxes_q = zeros(Float64, Nq, num_state_prognostic, dim)

#     local_flux = zeros(Float64, num_state_prognostic)
#     ∇ϕ = zeros(Float64, Nl, dim)
#     for iz = 1:Nz
#         for ix = 1:Nx
#             e = ix + (iz-1)*Nx

#             local_states_l = @view state_prognostic[:, :, e]
#             local_aux_q    = @view state_auxiliary_vol_q[:, :, e]

#             for s = 1:num_state_prognostic
#                 local_states_q[:, s] =  ϕl_q * local_states_l[:, s]
#             end

#             for iq = 1:Nq
#                 local_fluxes_q[iq, :, :] = flux_first_order(app, local_states_q[iq, :], local_aux_q[iq, :])
#             end

#             # Loop Legendre-Gauss-Lobatto points to construct ∇ϕ_j = ∇_ξϕ_j J⁻¹
#             for iq = 1:Nq
#                 M, ∂ξ∂x, ∂ξ∂z, ∂η∂x, ∂η∂z = vol_q_geo[:, iq, e]
#                 for il = 1:Nl

#                     tendency[il, :, e] .+= local_fluxes_q[iq, :, :] *( M * Dl_q[iq, il] * [∂ξ∂x ; ∂ξ∂z])
#                 end
#             end

#         end

#     end






#     vol_l_geo = mesh.vol_l_geo
#     Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
#     for iz = 1:Nz
#         for ix = 1:Nx
#             e = ix + (iz-1)*Nx

#             local_states_l = @view state_prognostic[:, :, e]
#             local_aux_l = @view state_auxiliary_vol_l[:, :, e]

#             for il = 1:Nl
#                 x, z, M = vol_l_geo[:, il, e]


#                 tendency[il, :, e] += source(app, local_states_l[il, :], local_aux_l[il, :]) * M 
#             end
#         end

#     end
# end

