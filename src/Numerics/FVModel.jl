#ghost cell, reconstruction for primitive variables
#  | ---- | ---- | ---- | ---- | ---- ...  | ---- |
# f1     f2     f3     f4    ...         f_Nz    f_{Nz+1}
#     c1     c2     c3                       c_Nz

"""
state_primitive, Δz
state_primitive = Array(num_state_prognostic, Nz), Δz = Array(Nz)
"""




function limiter(Δ⁻::Array{Float64,1}, Δ⁺::Array{Float64,1})
    Δ = zeros(size(Δ⁻))
    num_state = length(Δ⁻)
    for s = 1:num_state
        if Δ⁺[s] *  Δ⁻[s] > 0.0
            Δ[s] = 2 * Δ⁺[s] *  Δ⁻[s]/ (Δ⁺[s] + Δ⁻[s])
        end
    end
    return Δ
end

function reconstruction_1d_fv(app::Application, state_primitive_col, Δzc_col, 
    bc_bottom_type::String, bc_bottom_data::Union{Array{Float64, 1}, Nothing}, bc_bottom_n::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String, bc_top_data::Union{Array{Float64, 1}, Nothing}, bc_top_n::Union{Array{Float64, 1}, Nothing},
    state_primitive_face⁻::Array{Float64, 2}, state_primitive_face⁺::Array{Float64, 2})
    
    num_state_prognostic, Nz = size(state_primitive_col)
    
    
    ##########################################################################################################
    # compute face states by looping cells
    for iz = 1:Nz
        Δstate⁺ = state_primitive_col[:, mod1(iz+1,Nz)] - state_primitive_col[:, iz]
        Δstate⁻ = state_primitive_col[:, iz]   - state_primitive_col[:, mod1(iz-1,Nz)]
        Δz⁺ = (Δzc_col[mod1(iz+1,Nz)] + Δzc_col[iz])/2.0
        Δz⁻ = (Δzc_col[iz] + Δzc_col[mod1(iz-1,Nz)])/2.0
        
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        
        
        state_primitive_face⁺[:, iz]   = state_primitive_col[:, iz] - ∂state * Δzc_col[iz]/2.0
        state_primitive_face⁻[:, iz+1] = state_primitive_col[:, iz] + ∂state * Δzc_col[iz]/2.0
        
    end
    
    if bc_bottom_type == "periodic" && bc_top_type == "periodic"
        state_primitive_face⁻[:, 1] .=  state_primitive_face⁻[:, Nz+1]
        state_primitive_face⁺[:, Nz+1] .= state_primitive_face⁺[:, 1] 
        
        return ;
    end
    
    # update other type of boundary conditions
    if bc_bottom_type == "no-slip" || bc_bottom_type == "no-penetration"
        Δz1, Δz2 = Δzc_col[1], Δzc_col[2]
        # one-side extrapolation
        state_primitive_face⁺[:, 1] = (2*Δz1+Δz2)/(Δz1+Δz2)*state_primitive_col[:, 1] - Δz1/(Δz1+Δz2)*state_primitive_col[:, 2]
        state_primitive_face⁺[:, 1] = bc_impose(app, state_primitive_face⁺[:, 1], bc_bottom_type, bc_bottom_n)
        # interpolation
        
        Δstate⁺ = state_primitive_col[:, 2] - state_primitive_col[:, 1]
        Δstate⁻ = state_primitive_col[:, 1] - state_primitive_face⁺[:, 1]
        Δz⁺ = (Δz1 + Δz2)/2.0
        Δz⁻ = Δz1/2.0
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        state_primitive_face⁻[:, 2] = state_primitive_col[:, 1] + ∂state * Δz1/2.0
        
    else
        error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
    end
    
    # populate top ghost state
    if bc_top_type == "outlet"
        Δz1, Δz2 = Δzc_col[Nz], Δzc_col[Nz-1]
        # outlet state
        state_primitive_face⁺[:, Nz + 1] = bc_top_data
        
        Δstate⁺ = state_primitive_col[:, Nz] - state_primitive_col[:, Nz-1]
        Δstate⁻ = state_primitive_face⁺[:, Nz + 1] - state_primitive_col[:, Nz] 
        Δz⁺ = (Δz1+Δz2)/2.0
        Δz⁻ = Δz1/2.0
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        
        # one-side extrapolation
        state_primitive_face⁻[:, Nz + 1] = state_primitive_col[:, Nz] + ∂state * Δz1/2.0
        state_primitive_face⁺[:, Nz] = state_primitive_col[:, Nz] - ∂state * Δz1/2.0
    else
        error("bc_top_type = ", bc_top_type, " has not implemented")   
    end   
end


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
            
            bc_bottom_n = sgeo_v[1:2, il, 1, ix] 
            bc_top_n = sgeo_v[1:2, il, end, ix + (Nz - 1)*Nx] 
            reconstruction_1d_fv(app, state_primitive_col, Δzc_col, 
            bc_bottom_type, bc_bottom_data, bc_bottom_n, 
            bc_top_type, bc_top_data, bc_top_n, 
            state_primitive_face⁻, state_primitive_face⁺)
            
            
            for iz = 1:Nz+1
                e⁺ =  ix + (iz-1)*Nx
                loc_aux = (iz == Nz+1 ? state_auxiliary_surf_v[il, :,  end, ix + (iz-2)*Nx] : state_auxiliary_surf_v[il, :,  1, e⁺])
                state_prognostic_face⁻[:,iz] = prim_to_prog(app, state_primitive_face⁻[:,iz], loc_aux)
                state_prognostic_face⁺[:,iz] = prim_to_prog(app, state_primitive_face⁺[:,iz], loc_aux)
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
                    elseif bc_bottom_type == "no-slip" || bc_bottom_type == "no-penetration"
                        
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
                    if bc_top_type == "periodic"
                        # use the auxiliary state in  local_aux⁻
                        local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, state_prognostic_face⁺[:, iz], local_aux⁻, [n1;n2])
                        
                    elseif bc_top_type == "outlet"
                        
                        # use the auxiliary state in  local_aux⁻
                        local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, state_prognostic_face⁺[:, iz], local_aux⁻, [n1;n2])
                        
                        
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


