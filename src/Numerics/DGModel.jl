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
    
    num_state_prognostic = app.num_state_prognostic
    dim, vol_q_geo, ϕl_q, Dl_q = mesh.dim, mesh.vol_q_geo, mesh.ϕl_q, mesh.Dl_q
    
    
    # reconstructed local state at quadrature points
    local_states_q = zeros(Float64, Nq, num_state_prognostic)
    local_fluxes_q = zeros(Float64, Nq, num_state_prognostic, dim)
    
    local_flux = zeros(Float64, num_state_prognostic)
    ∇ϕ = zeros(Float64, Nl, dim)
    for iz = 1:Nz
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
                    # @info size(tendency), size(local_fluxes_q), size(Dl_q)
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
    sgeo_h = mesh.sgeo_h
    
    for iz = 1:Nz
        # Consider the internal face 
        for ix = 1:Nx
            e⁺ =   ix + (iz-1)*Nx
            # periodic boundary condition
            e⁻ = mod1(ix - 1, Nx) + (iz-1)*Nx
            
            local_state⁻ = state_prognostic[end, :, e⁻]
            local_state⁺ = state_prognostic[1,   :, e⁺]
            
            local_aux⁻ = state_auxiliary_surf_h[1, :, end, e⁻]
            local_aux⁺ = state_auxiliary_surf_h[1, :, 1, e⁺]
            
            
            (n1, n2, sM) = sgeo_h[:, 1, end, e⁻] 
            
            local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁺, [n1;n2])
            
            tendency[1,   :, e⁺] .+= sM * local_flux
            tendency[end, :, e⁻] .-= sM * local_flux
            
            
        end
    end
end



#ghost cell, reconstruction for primitive variables
function vertical_interface_tendency!(
    app::Application,
    mesh::Mesh,
    state_prognostic::Array{Float64, 3},
    state_auxiliary_surf_v::Array{Float64,4},
    tendency::Array{Float64, 3}
    )
    
    sgeo_v = mesh.sgeo_v
    
    for ix = 1:Nx
        for il = 1:Nl

            for iz = 1:Nz+1
                
                # internal faces
                if iz >= 2 && iz <= Nz
                    
                    # top element
                    e⁺ =  ix + (iz-1)*Nx
                    # bottom element
                    e⁻ =  ix + (iz-2)*Nx
                    
                    local_state⁺ = state_prognostic[il, :, e⁺]
                    local_state⁻ = state_prognostic[il, :, e⁻]
                    
                    local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                    local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                    
                    (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                    
                    local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁺, [n1;n2])
                    
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    tendency[il, :,  e⁺]  .+=  sM * local_flux
                    
                    
                else # top and bottom boundary condition
                    
                    
                    bc_top_type, bc_bottom_type = app.bc_top_type,  app.bc_bottom_type
                    # bottom boundary condition
                    if iz == 1
                        
                        # top element
                        e⁺ =  ix + (iz-1)*Nx
                        
                        local_state⁺ = state_prognostic[il, :, e⁺]
                        
                        (n1, n2, sM) = sgeo_v[:, il, 1, e⁺] 
                        
                        if bc_bottom_type == "periodic"
                            e⁻ =  ix + (Nz-1)*Nx
                            
                            local_state⁻ = state_prognostic[il, :, e⁻]
                            
                            local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                            local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                            
                            local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁺, [-n1;-n2])
                        else
                            
                            error("bc_bottom_type = ", bc_bottom_type, " has not implemented")
                            
                        end
                        
                        
                        tendency[il, :, e⁺]  .+=  sM * local_flux
                        
                        
                    else    # top boundary condition
                        @assert(iz == Nz + 1)
                        
                        
                        # bottom element
                        e⁻ =  ix + (iz-2)*Nx
                        
                        
                        local_state⁻ = state_prognostic[il, :, e⁻]
                        (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                        
                        if bc_top_type == "periodic"
                            
                            e⁺ =  ix
                            
                            local_state⁺ = state_prognostic[il, :, e⁺]
                            
                            local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                            local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                            
                            local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻,  local_state⁺, local_aux⁺, [n1;n2])
                        else
                            
                        end
                        tendency[il, :, e⁻]  .-=  sM * local_flux
                        
                    end
                end
                
            end 
        end
    end
end




#ghost cell, reconstruction for primitive variables
function hydrostatic_vertical_interface_tendency!(
    app::Application,
    mesh::Mesh,
    state_prognostic::Array{Float64, 3},
    state_auxiliary_surf_v::Array{Float64,4},
    tendency::Array{Float64, 3}
    )
    
    sgeo_v = mesh.sgeo_v
    
    for ix = 1:Nx
        for il = 1:Nl
            
            for iz = 1:Nz+1
                
                # internal faces
                if iz >= 2 && iz <= Nz
                    
                    # top element
                    e⁺ =  ix + (iz-1)*Nx
                    # bottom element
                    e⁻ =  ix + (iz-2)*Nx
                    
                    local_state⁺ = state_prognostic[il, :, e⁺]
                    local_state⁻ = state_prognostic[il, :, e⁻]
                    
                    local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                    local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                    
                    (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                    
                    local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁺, [n1;n2])
                    
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    tendency[il, :,  e⁺]  .+=  sM * local_flux
                    
                    
                else # top and bottom boundary condition
                    
                    
                    bc_top_type, bc_bottom_type = app.bc_top_type,  app.bc_bottom_type
                    # bottom boundary condition
                    if iz == 1
                        
                        # top element
                        e⁺ =  ix + (iz-1)*Nx
                        
                        local_state⁺ = state_prognostic[il, :, e⁺]
                        
                        (n1, n2, sM) = sgeo_v[:, il, 1, e⁺] 
                        
                        if bc_bottom_type == "periodic"
                            e⁻ =  ix + (Nz-1)*Nx
                            
                            local_state⁻ = state_prognostic[il, :, e⁻]
                            
                            local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                            local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                            
                            local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁺, [-n1;-n2])
                        else
                            
                            error("bc_bottom_type = ", bc_bottom_type, " has not implemented")
                            
                        end
                        
                        
                        tendency[il, :, e⁺]  .+=  sM * local_flux
                        
                        
                    else    # top boundary condition
                        @assert(iz == Nz + 1)
                        
                        
                        # bottom element
                        e⁻ =  ix + (iz-2)*Nx
                        
                        
                        local_state⁻ = state_prognostic[il, :, e⁻]
                        (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                        
                        if bc_top_type == "periodic"
                            
                            e⁺ =  ix
                            
                            local_state⁺ = state_prognostic[il, :, e⁺]
                            
                            local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                            local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                            
                            local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻,  local_state⁺, local_aux⁺, [n1;n2])
                        else
                            
                        end
                        tendency[il, :, e⁻]  .-=  sM * local_flux
                        
                    end
                end
                
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
    for iz = 1:Nz
        for ix = 1:Nx
            e = ix + (iz-1)*Nx
            
            local_states_l = @view state_prognostic[:, :, e]
            local_aux_l = @view state_auxiliary_vol_l[:, :, e]
            
            for il = 1:Nl
                x, z, M = vol_l_geo[:, il, e]
                tendency[il, :, e] += source(app, local_states_l[il, :], local_aux_l[il, :]) * M
            end
            
            
        end
        
    end
end

