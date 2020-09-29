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
    
    Nx, Nz = mesh.Nx, mesh.Nz
    bc_left_type, bc_left_data = app.bc_left_type, app.bc_left_data
    bc_right_type, bc_right_data = app.bc_left_type, app.bc_right_data
    
    sgeo_h = mesh.sgeo_h
    
    for iz = 1:Nz
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
                    
                elseif bc_left_data == "no-slip"
                    
                    local_flux = wall_flux_first_order(app, local_state⁺, local_aux⁺, [n1;n2])
                       
                else
                    error("bc_left_type = ", bc_left_type, " has not implemented")   
                end
                
                tendency[1,   :, e⁺] .+= sM * local_flux
                
            elseif ix == Nx + 1 # right bc
                
                (n1, n2, sM) = sgeo_h[:, 1, end, e⁻] 
                
                if bc_right_type == "periodic"
                    local_flux = numerical_flux_first_order(app, local_state⁻, local_aux⁻, local_state⁺, local_aux⁻, [n1;n2])
                    
                elseif bc_right_data == "no-slip"
                    
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

        # @info tendency[:, 1, :]
        # error("stop ")
    end
end


function limiter(Δ⁻::Array{Float64,1}, Δ⁺::Array{Float64,1})
    Δ = zeros(size(Δ⁻))
    num_state = length(Δ⁻)
    for s = 1:num_state
        if Δ⁺[s] *  Δ⁻[s] > 0.0
            Δ[s] = Δ⁺[s] *  Δ⁻[s]/ (Δ⁺[s] + Δ⁻[s])
        end
    end
    
    return Δ
    
end


#ghost cell, reconstruction for primitive variables
#  | ---- | ---- | ---- | ---- | ---- ...  | ---- |
# f1     f2     f3     f4    ...         f_Nz    f_{Nz+1}
#     c1     c2     c3                       c_Nz
function vertical_interface_tendency!(
    app::Application,
    mesh::Mesh,
    state_primitive::Array{Float64, 3},
    state_auxiliary_surf_v::Array{Float64,4},
    tendency::Array{Float64, 3}
    )
    dim = 2
    sgeo_v = mesh.sgeo_v
    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
    num_state_prognostic = app.num_state_prognostic

    bc_bottom_type, bc_bottom_data = app.bc_bottom_type, app.bc_bottom_data
    bc_top_type, bc_top_data = app.bc_top_type, app.bc_top_data
    
    
    state_primitive_face⁺  = zeros(Float64, num_state_prognostic, Nz+1)
    state_primitive_face⁻  = zeros(Float64, num_state_prognostic, Nz+1)
    state_prognostic_face⁺ = zeros(Float64, num_state_prognostic, Nz+1)
    state_prognostic_face⁻ = zeros(Float64, num_state_prognostic, Nz+1)
    ghost_state⁺ = zeros(Float64, num_state_prognostic)
    ghost_state⁻ = zeros(Float64, num_state_prognostic)
    
    
    for ix = 1:Nx
        for il = 1:Nl
            # single colume treatment 
            
            
            ##########
            #  reconstruct interface states by looping each cell, construct the bottom and top states
            Δzc_col = @view mesh.Δzc[il, ix, :]
            state_primitive_col = @view state_primitive[il, : , ix:Nx:end]
            
            # populate bottom ghost state
            if bc_bottom_type == "periodic"
                ghost_state⁻ .= state_primitive_col[:, Nz]
                ghost_Δz⁻ = Δzc_col[Nz]
            elseif bc_bottom_type == "no-slip"
                ghost_state⁻ .= [state_primitive_col[1, 1]; -state_primitive_col[2, 1] ; -state_primitive_col[2, 1] ; state_primitive_col[dim+2, 1]]
                ghost_Δz⁻ = Δzc_col[1]
            else
                error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
            end
            
            # populate top ghost state
            if bc_top_type == "periodic"
                ghost_state⁺ .= state_primitive_col[:, 1]
                ghost_Δz⁺ = Δzc_col[1]
            elseif bc_top_type == "no-slip"
                ghost_state⁺ .= [state_primitive_col[1, Nz]; -state_primitive_col[2, Nz] ; -state_primitive_col[2, Nz] ; state_primitive_col[dim+2, Nz]]
                ghost_Δz⁺ = Δzc_col[Nz]
            else
                error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
            end
            
            ##########################################################################################################
            # compute face states by looping cells
            for iz = 1:Nz
                Δstate⁺ = (iz==Nz ? ghost_state⁺ : state_primitive_col[:, iz+1]) - state_primitive_col[:, iz]
                Δstate⁻ = state_primitive_col[:, iz]   - (iz==1 ? ghost_state⁻ : state_primitive_col[:, iz-1])
                Δz⁺ = ((iz==Nz ? ghost_Δz⁺ : Δzc_col[iz+1]) + Δzc_col[iz])/2.0
                Δz⁻ = (Δzc_col[iz] + (iz==1 ? ghost_Δz⁻ : Δzc_col[iz-1]))/2.0
                
                ∂state = limiter(Δstate⁻/Δz⁻, Δstate⁺/Δz⁺)
                state_primitive_face⁺[:, iz]   = state_primitive_col[:, iz] - ∂state * Δzc_col[iz]/2.0
                state_primitive_face⁻[:, iz+1] = state_primitive_col[:, iz] + ∂state * Δzc_col[iz]/2.0
            end
            
            if bc_bottom_type == "periodic"
                state_primitive_face⁻[:, 1] .=  state_primitive_face⁻[:, Nz+1]
            else
                # should not use it 
                state_primitive_face⁻[:, 1] .= NaN64
            end
            
            if bc_top_type == "periodic"
                
                state_primitive_face⁺[:, Nz+1] .= state_primitive_face⁺[:, 1] 
            else
                # should not use it 
                state_primitive_face⁺[:, Nz+1] .= NaN64
            end
            
            
            for iz = 1:Nz+1
                e⁺ =  ix + (iz-1)*Nx
                loc_aux = (iz == Nz+1 ? state_auxiliary_surf_v[il, :,  end, ix + (iz-2)*Nx] : state_auxiliary_surf_v[il, :,  1, e⁺])
                state_primitive_face⁻[:,iz] = prim_to_prog(app, state_primitive_face⁻[:,iz], loc_aux)
                state_primitive_face⁺[:,iz] = prim_to_prog(app, state_primitive_face⁺[:,iz], loc_aux)
            end
            ##########################################################################################################
            # compute face flux 
            
            
            
            # loop face 
            for iz = 1:Nz+1
                # face iz ;  bottom cell iz-1 ; top cell is iz
                # bottom 
                if iz == 1
                    e⁺ =  ix + (iz-1)*Nx
                    local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺]
                    
                    (n1, n2, sM) = sgeo_v[:, il, 1, e⁺] 
                    # the normal points from e⁻ to e⁺
                    n1, n2 = -n1, -n2
                    if bc_bottom_type == "periodic"
                        # use the auxiliary state in  local_aux⁺
                        local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁺, state_prognostic_face⁺[:, iz], local_aux⁺, [n1;n2])
                    elseif bc_bottom_type == "no-slip"
                        
                        local_flux = wall_flux_first_order(app, state_prognostic_face⁺[:, iz], local_aux⁺, [n1;n2])
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
                    end
                    
                    tendency[il, :,  e⁺]  .+=  sM * local_flux
                    
                    # top 
                elseif iz == Nz+1
                    e⁻ =  ix + (iz-2)*Nx  
                    local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻] 
                    (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                    if bc_bottom_type == "periodic"
                        # use the auxiliary state in  local_aux⁻
                        local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, state_prognostic_face⁺[:, iz], local_aux⁻, [n1;n2])
                        
                    elseif bc_top_type == "no-slip"
                        
                        local_flux = wall_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, [n1;n2])
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
                    end
                    
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    
                else
                    # bottom element
                    e⁻ =  ix + (iz-2)*Nx   
                    e⁺ =  ix + (iz-1)*Nx  
                    local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻]
                    local_aux⁺ = state_auxiliary_surf_v[il, :,  1, e⁺] 
                    
                    (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                    
                    local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, state_prognostic_face⁺[:, iz], local_aux⁺, [n1;n2])
                    
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    tendency[il, :,  e⁺]  .+=  sM * local_flux
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
    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
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

