include("WENOModel.jl")

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

function fv_recon(h::Array{Float64, 1}, u::Array{Float64, 2})
    num_state_prognostic = size(u, 1)
    Δz⁻, Δz, Δz⁺ = h
    

    Δu⁺ = (u[:, 3] - u[:, 2])
    Δu⁻ =  (u[:, 2] - u[:, 1])
    
    # @info iz, Δstate⁺, Δstate⁻
    # @info state_primitive_col[:, mod1(iz-1,Nz)], state_primitive_col[:, iz], state_primitive_col[:, mod1(iz+1,Nz)]
    ∂state = 2.0*limiter(Δu⁺/(Δz⁺ + Δz), Δu⁻/(Δz⁻ + Δz))
    u⁺  = u[:,2]  + ∂state * Δz/2.0
    u⁻  = u[:,2]  - ∂state * Δz/2.0
    
    return u⁻, u⁺
end

"""
h1     h2     h3    
|--i-1--|--i--|--i+1--|
without hat : i - 1/2
with hat    : i + 1/2  
"""
# function fv2_recon(app::Application, h::Array{Float64, 1}, u::Array{Float64, 2})
#     num_state_prognostic = size(u, 1)
#     h1, h2, h3 = h
#     ρ1, ρ2, ρ3 = u[1, :]
#     p1, p2, p3 = u[4, :]
#     p1_ref, p2_ref, p3_ref =  p2, p2 - (ρ2+ρ3)
#     p_ref = 

#     # at  i - 1/2,  i + 1/2
#     u⁻ = zeros(Float64, num_state_prognostic)
#     u⁺ = zeros(Float64, num_state_prognostic)
#     for i = 1:num_state_prognostic
#         u⁻[i] += w[i, :]' * P[i, 1, :]
#         u⁺[i] += ŵ[i, :]' * P[i, 2, :]
#     end

#     return u⁻, u⁺

# end



function reconstruction_1d_fv(app::Application, state_primitive_col, Δzc_col, 
    state_primitive_face⁻::Array{Float64, 2}, state_primitive_face⁺::Array{Float64, 2})
    
    num_state_prognostic, Nz = size(state_primitive_col)
    
    
    num_left_stencil = 1
    state_primitive_fv, Δz_fv = zeros(num_state_prognostic, 2num_left_stencil+1), zeros(2num_left_stencil+1)
    ##########################################################################################################
    # compute face states by looping cells
    for iz = 1:Nz
        
        for is = 1: 2num_left_stencil+1
            state_primitive_fv[:, is] = state_primitive_col[:, mod1(iz - num_left_stencil + is - 1, Nz)]
            Δz_fv[is] =  Δzc_col[mod1(iz - num_left_stencil + is - 1, Nz)]
        end


        if app.hydrostatic_balance
            g = app.g
            ρ, p, Δz = state_primitive_col[1, iz],  state_primitive_col[4, iz],  Δzc_col[iz]
            # bottom face⁺ and top face⁻
            p_face⁺, p_face⁻ = p + ρ*Δz*g/2.0, p - ρ*Δz*g/2.0
            

            
            
            # subtract the hydrostatic balance p_ref
            state_primitive_fv[4, 1] -= p + g*(state_primitive_fv[1, 1]*Δz_fv[1] + ρ*Δz)/2.0
            state_primitive_fv[4, 2] -= p
            state_primitive_fv[4, 3] -= p - g*(state_primitive_fv[1, 3]*Δz_fv[3] + ρ*Δz)/2.0
            
        end
        
        # #face:          iz      iz+1
        # #cell: |  iz-1   |   iz   |  iz+1  |
        # Δz⁻, Δz, Δz⁺ = Δzc_col[mod1(iz-1,Nz)], Δzc_col[iz], Δzc_col[mod1(iz+1,Nz)]
        
        # state_primitive0  .= state_primitive_col[:, iz]
        
        # state_primitive0⁻ .= state_primitive0 ; 
        # state_primitive0⁻[4] = state_primitive0[4] + g*(state_primitive0[1]*Δz + state_primitive_col[1, mod1(iz-1,Nz)]*Δz⁻)/2.0
        # state_primitive_face⁺[:, iz] .= state_primitive0
        # state_primitive_face⁺[4, iz]  = state_primitive0[4] + g*(state_primitive0[1]*Δz)/2.0
        
        # state_primitive0⁺ .= state_primitive0 ; 
        # state_primitive0⁺[4] = state_primitive0[4] - g*(state_primitive0[1]*Δz + state_primitive_col[1, mod1(iz+1,Nz)]*Δz⁺)/2.0
        # state_primitive_face⁻[:, iz+1] .= state_primitive0
        # state_primitive_face⁻[4, iz+1]  = state_primitive0[4] - g*(state_primitive0[1]*Δz)/2.0
        
        # Δstate⁺ = (state_primitive_col[:, mod1(iz+1,Nz)] - state_primitive0⁺)
        # Δstate⁻ =                                                             - (state_primitive_col[:, mod1(iz-1,Nz)] - state_primitive0⁻)
        
        # # @info iz, Δstate⁺, Δstate⁻
        # # @info state_primitive_col[:, mod1(iz-1,Nz)], state_primitive_col[:, iz], state_primitive_col[:, mod1(iz+1,Nz)]
        # ∂state = 2.0*limiter(Δstate⁺/(Δz⁺ + Δz), Δstate⁻/(Δz⁻ + Δz))
        # state_primitive_face⁺[:, iz]   .-=  ∂state * Δz/2.0
        # state_primitive_face⁻[:, iz+1] .+=  ∂state * Δz/2.0
        
        (state_primitive_face⁺[:, iz], state_primitive_face⁻[:, iz+1]) = fv_recon(Δz_fv, state_primitive_fv)
        
        if app.hydrostatic_balance
            # add the hydrostatic balance p_ref
            state_primitive_face⁺[4, iz]   += p_face⁺
            state_primitive_face⁻[4, iz+1] += p_face⁻
        end
        
    end
    
    # error("stop")
    
end


function reconstruction_1d(app::Application, method::String, state_primitive_col, Δzc_col, 
    state_auxiliary_vol_l_col, state_auxiliary_surf_v_col,
    bc_bottom_type::String, bc_bottom_data::Union{Array{Float64, 1}, Nothing}, bc_bottom_n::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String, bc_top_data::Union{Array{Float64, 1}, Nothing}, bc_top_n::Union{Array{Float64, 1}, Nothing},
    state_primitive_face⁻::Array{Float64, 2}, state_primitive_face⁺::Array{Float64, 2})
    
    Nz = length(Δzc_col)
    
    
    if method == "FV"
        reconstruction_1d_fv(app, state_primitive_col, Δzc_col, 
        state_primitive_face⁻, state_primitive_face⁺)
    elseif method == "WENO3"
        reconstruction_1d = reconstruction_1d_weno3(app, state_primitive_col, Δzc_col, 
        state_primitive_face⁻, state_primitive_face⁺)
    elseif method == "WENO5"
        reconstruction_1d = reconstruction_1d_weno5(app, state_primitive_col, Δzc_col, 
        state_primitive_face⁻, state_primitive_face⁺)
    else
        error("vertical method : ", method, " has not implemented")
    end
    
    
    if bc_bottom_type == "periodic" && bc_top_type == "periodic"
        state_primitive_face⁻[:, 1] .=  state_primitive_face⁻[:, Nz+1]
        state_primitive_face⁺[:, Nz+1] .= state_primitive_face⁺[:, 1] 
    else
        
        g = app.g
        # update other type of boundary conditions
        if bc_bottom_type == "no-slip" || bc_bottom_type == "no-penetration"
            
            Δz, Δz⁺ = Δzc_col[1], Δzc_col[2]
            
            
            state_primitive0  = state_primitive_col[:, 1]
            
            # state_primitive0⁺ .= state_primitive_col[:, 2] ; 
            # state_primitive0⁺[4] = state_primitive0[4] - g*(state_primitive0[1]*Δz + state_primitive0⁺[1]*Δz⁺)/2.0
            
            
            #constant reconstruction
            
            state_primitive_face⁺[:, 1] .= state_primitive0;
            state_primitive_face⁺[4, 1]  = state_primitive0[4] + g*state_primitive0[1]*Δz/2.0
            state_primitive_face⁺[:, 1] .= bc_impose(app, state_primitive_face⁺[:,1], bc_bottom_type, bc_bottom_n)
            
            # one-side extrapolation vs central 
            state_primitive_face⁻[:, 2].= state_primitive0;
            
            # state_primitive_face⁻[:, 2].=  Δz⁺/(Δz + Δz⁺)*state_primitive0 + Δz/(Δz + Δz⁺)*state_primitive_col[:, 2];
            
            state_primitive_face⁻[4, 2] = state_primitive0[4] - g*state_primitive0[1]*Δz/2.0
            
            
        else
            error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
        end
        
        
        
        # populate top ghost state
        if bc_top_type == "outlet"
            
            
            Δz, Δz⁻ = Δzc_col[Nz], Δzc_col[Nz - 1]
            
            state_primitive0  = state_primitive_col[:, Nz]
            
            #constant reconstruction one-side extrapolation
            
            state_primitive_face⁺[:, Nz] .= state_primitive0;
            state_primitive_face⁺[4, Nz]  = state_primitive0[4] + g*state_primitive0[1]*Δz/2.0
            
            # one-side extrapolation vs central 
            state_primitive_face⁻[:, Nz+1] .= state_primitive0;
            # state_primitive_face⁻[:, Nz+1] .= Δz⁻/(Δz + Δz⁻)*state_primitive0 + Δz/(Δz + Δz⁻)*state_primitive_col[:, Nz-1];
            state_primitive_face⁻[4, Nz+1]  = state_primitive0[4] - g*state_primitive0[1]*Δz/2.0
            
            
            
            # outlet state
            state_primitive_face⁺[:, Nz + 1] = bc_top_data
            
            # Δstate⁺ = state_primitive_col[:, Nz] - state_primitive_col[:, Nz-1]
            # Δstate⁻ = state_primitive_face⁺[:, Nz + 1] - state_primitive_col[:, Nz] 
            # Δz⁺ = (Δz1+Δz2)/2.0
            # Δz⁻ = Δz1/2.0
            # ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
            
            # # one-side extrapolation
            # state_primitive_face⁻[:, Nz + 1] = state_primitive_col[:, Nz] + ∂state * Δz1/2.0
            # state_primitive_face⁺[:, Nz] = state_primitive_col[:, Nz] - ∂state * Δz1/2.0
        elseif bc_top_type == "no-slip" || bc_top_type == "no-penetration"
            
            Δz, Δz⁻ = Δzc_col[Nz], Δzc_col[Nz - 1]
            
            state_primitive0  = state_primitive_col[:, Nz]
            
            #constant reconstruction one-side extrapolation
            
            #one-side extrapolation vs central 
            state_primitive_face⁺[:, Nz] .= state_primitive0;
            # state_primitive_face⁺[:, Nz] .= Δz⁻/(Δz + Δz⁻)*state_primitive0 + Δz/(Δz + Δz⁻)*state_primitive_col[:, Nz-1];
            state_primitive_face⁺[4, Nz]  = state_primitive0[4] + g*state_primitive0[1]*Δz/2.0
            
            
            state_primitive_face⁻[:, Nz+1] .= state_primitive0;
            state_primitive_face⁻[4, Nz+1]  = state_primitive0[4] - g*state_primitive0[1]*Δz/2.0
            state_primitive_face⁻[:, Nz+1] = bc_impose(app, state_primitive_face⁻[:, Nz+1], bc_top_type, bc_top_n)
            
            
        else
            error("bc_top_type = ", bc_top_type, " has not implemented")   
        end 
        
    end
    
end



function vertical_interface_first_order_tendency!(
    app::Application,
    mesh::Mesh,
    state_primitive::Array{Float64, 3},
    state_auxiliary_vol_l::Array{Float64,3},
    state_auxiliary_surf_v::Array{Float64,4},
    tendency::Array{Float64, 3};
    method::String
    )
    
    
    dim = 2
    sgeo_v = mesh.sgeo_v
    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
    num_state_prognostic = app.num_state_prognostic
    
    bc_bottom_type, bc_bottom_data = app.bc_bottom_type, app.bc_bottom_data
    bc_top_type, bc_top_data = app.bc_top_type, app.bc_top_data
    
    
    
    
    
    
    Threads.@threads for ix = 1:Nx
        state_primitive_face⁺  = zeros(Float64, num_state_prognostic, Nz+1)
        state_primitive_face⁻  = zeros(Float64, num_state_prognostic, Nz+1)
        state_prognostic_face⁺ = zeros(Float64, num_state_prognostic, Nz+1)
        state_prognostic_face⁻ = zeros(Float64, num_state_prognostic, Nz+1)
        
        for il = 1:Nl
            # single colume treatment  
            ##########
            #  reconstruct interface states by looping each cell, construct the bottom and top states
            id_col = ix:Nx:Nz*Nx
            Δzc_col = @view mesh.Δzc[il, ix, :]
            state_primitive_col = @view state_primitive[il, : , id_col]
            
            # normal 
            bc_bottom_n = sgeo_v[1:2, il, 1, ix] 
            bc_top_n = sgeo_v[1:2, il, end, ix + (Nz - 1)*Nx] 
            
            
            state_auxiliary_surf_v_col = @view state_auxiliary_surf_v[il,  :, :, id_col]
            state_auxiliary_vol_l_col = @view state_auxiliary_vol_l[il, :, id_col]
            
            reconstruction_1d(app, method, 
            state_primitive_col, Δzc_col,
            state_auxiliary_vol_l_col, state_auxiliary_surf_v_col,
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
                    
                    # @info iz, sM * local_flux
                    
                    
                    # top 
                elseif iz == Nz+1
                    e⁻ =  ix + (Nz-1)*Nx  
                    local_aux⁻ = state_auxiliary_surf_v[il, :,  end, e⁻] 
                    (n1, n2, sM) = sgeo_v[:, il, end, e⁻] 
                    if bc_top_type == "periodic"
                        # use the auxiliary state in  local_aux⁻
                        local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, state_prognostic_face⁺[:, iz], local_aux⁻, [n1;n2])
                        
                    elseif bc_top_type == "outlet"
                        
                        # use the auxiliary state in  local_aux⁻
                        local_flux = numerical_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, state_prognostic_face⁺[:, iz], local_aux⁻, [n1;n2])
                        
                        # @info state_prognostic_face⁻[:, iz],  state_prognostic_face⁺[:, iz]
                        
                    elseif bc_bottom_type == "no-slip" || bc_bottom_type == "no-penetration"
                        
                        local_flux = wall_flux_first_order(app, state_prognostic_face⁻[:, iz], local_aux⁻, [n1;n2]) 
                        
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
                    end
                    
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    
                    
                    # @info iz, sM * local_flux
                    
                    
                    
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


function gradient_correction(∇state_gradient⁻::Array{Float64, 2}, state_gradient⁻::Array{Float64, 1}, Δz⁻::Float64,  
                             ∇state_gradient⁺::Array{Float64, 2}, state_gradient⁺::Array{Float64, 1}, Δz⁺::Float64, ds::Array{Float64, 1})
    ∇state_gradient = (∇state_gradient⁻*Δz⁺ + ∇state_gradient⁺*Δz⁻)/(Δz⁺ + Δz⁻)
    δstate_gradient = (state_gradient⁺ - state_gradient⁻)
    ds_square = ds[1]*ds[1] + ds[2]*ds[2]
    return ∇state_gradient .+ (δstate_gradient - ∇state_gradient*ds) * (ds'/ds_square)
end

function vertical_interface_second_order_tendency!(
    app::Application,
    mesh::Mesh,
    state_primitive::Array{Float64, 3},
    state_gradient::Array{Float64, 3},
    ∇state_gradient::Array{Float64, 4},
    state_auxiliary_vol_l::Array{Float64,3},
    state_auxiliary_surf_v::Array{Float64,4},
    tendency::Array{Float64, 3}
    )
    
    
    dim = 2
    sgeo_v, vol_l_geo = mesh.sgeo_v, mesh.vol_l_geo
    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
    
    bc_bottom_type, bc_bottom_data = app.bc_bottom_type, app.bc_bottom_data
    bc_top_type, bc_top_data = app.bc_top_type, app.bc_top_data

    Threads.@threads for ix = 1:Nx        
        for il = 1:Nl
            # single colume treatment  
            
            Δzc_col = @view mesh.Δzc[il, ix, :]
            
            # loop face 
            for iz = 1:Nz+1
                e⁻, e⁺ =  ix+mod(iz-2, Nz)*Nx, ix+mod(iz-1, Nz)*Nx 
                Δz⁻, Δz⁺ = Δzc_col[mod1(iz - 1, Nz)], Δzc_col[mod1(iz, Nz)]
                state_gradient⁻, state_gradient⁺   = state_gradient[il, :, e⁻], state_gradient[il, :, e⁺]
                ∇state_gradient⁻, ∇state_gradient⁺ = ∇state_gradient[il, :, e⁻, :], ∇state_gradient[il, :, e⁺, :]
                local_aux⁻, local_aux⁺ = state_auxiliary_surf_v[il, :,  end, e⁻], state_auxiliary_surf_v[il, :,  1, e⁺]
                x⁺, z⁺ = vol_l_geo[1:2, il ,e⁺]
                x⁻, z⁻ = vol_l_geo[1:2, il ,e⁻]
                
                (n1, n2) = (iz == 1 ? -sgeo_v[1:2, il, 1, e⁺] : sgeo_v[1:2, il, end, e⁻])
                sM = (iz == 1 ? sgeo_v[3, il, 1, e⁺] : sgeo_v[3, il, end, e⁻])
        
                
                
                # face iz ;  bottom cell iz-1 ; top cell is iz
                # bottom 
                if iz == 1
                    
                    x,  z =  sgeo_v[4:5, il , 1, e⁺]
                    xp,  zp =  sgeo_v[4:5, il , end, e⁻]
                    if bc_bottom_type == "periodic"
                        state_primitive_face = (state_primitive[il, :, e⁻]*Δz⁺ + state_primitive[il, :, e⁺]*Δz⁻)/(Δz⁺ + Δz⁻)
                        ∇state_gradient_face =  gradient_correction(∇state_gradient⁻, state_gradient⁻, Δz⁻,  ∇state_gradient⁺, state_gradient⁺, Δz⁺, [x⁺-x+xp-x⁻, z⁺-z+zp-z⁻])
                        # use the auxiliary state in  local_aux⁺
                        local_flux = flux_second_order_prim(app, state_primitive_face, ∇state_gradient_face, local_aux⁺)*[n1;n2]
                    elseif bc_bottom_type == "no-penetration"
                        # prescribed τ⋅n, the entraining flux is ρ * τ⋅n
                        # constant reconstruction for the primitive variables, for we only need ρ, no hydrostatic balanced reconstruction is needed
                        state_primitive_face = state_primitive[il, :, e⁺]
                        # the flux is in the z+ direction
                        local_flux = bc_second_order_flux(app, state_primitive_face, bc_bottom_type, bc_bottom_data)
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " for second order vertical flux has not implemented")   
                    end
                    
                    tendency[il, :,  e⁺]  .+=  sM * local_flux
                    
                    # top 
                elseif iz == Nz+1

                    x,  z =  sgeo_v[4:5, il , end, e⁻]
                    xp,  zp =  sgeo_v[4:5, il , 1, e⁺]
                    if bc_top_type == "periodic"
                        state_primitive_face = (state_primitive[il, :, e⁻]*Δz⁺ + state_primitive[il, :, e⁺]*Δz⁻)/(Δz⁺ + Δz⁻)
                        ∇state_gradient_face =  gradient_correction(∇state_gradient⁻, state_gradient⁻, Δz⁻,  ∇state_gradient⁺, state_gradient⁺, Δz⁺, [x-x⁻+x⁺-xp, z-z⁻+z⁺-zp])
                        # use the auxiliary state in  local_aux⁻
                        local_flux = flux_second_order_prim(app, state_primitive_face, ∇state_gradient_face, local_aux⁻)*[n1;n2]
                    elseif bc_top_type == "no-penetration"
                        # prescribed flux 
                        # constant reconstruction for the primitive variables, for we only need ρ, no hydrostatic balanced reconstruction is needed
                        state_primitive_face = state_primitive[il, :, e⁻]
                        # the flux is in the z+ direction
                        local_flux = bc_second_order_flux(app, state_primitive_face, bc_top_type, bc_top_data)
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " for second order vertical flux has not implemented")   
                    end
                    
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    
                else
 
        
                      
                    
                    state_primitive_face = (state_primitive[il, :, e⁻]*Δz⁺ + state_primitive[il, :, e⁺]*Δz⁻)/(Δz⁺ + Δz⁻)
                    ∇state_gradient_face =  gradient_correction(∇state_gradient⁻, state_gradient⁻, Δz⁻,  ∇state_gradient⁺, state_gradient⁺, Δz⁺, [x⁺-x⁻, z⁺-z⁻])
                    local_flux = flux_second_order_prim(app, state_primitive_face, ∇state_gradient_face, local_aux⁻)*[n1;n2]
                    
        
                    tendency[il, :,  e⁻]  .-=  sM * local_flux
                    tendency[il, :,  e⁺]  .+=  sM * local_flux

                end
            end 
            
        end
    end
end



"""
This function compute ∂Y/∂η
This function must be called after 
    horizontal_interface_gradient_tendency!
    horizontal_volume_gradient_tendency!
"""
function vertical_gradient_tendency!(
    app::Application,
    mesh::Mesh,
    state_gradient::Array{Float64, 3},
    ∇ref_state_gradient::Array{Float64, 4}
    )  
    
    Nx, Nz, Nl = mesh.Nx, mesh.Nz, mesh.Nl
    num_state_gradient = app.num_state_gradient
    bc_bottom_type, bc_top_type = app.bc_bottom_type, app.bc_top_type


    Threads.@threads for ix = 1:Nx
        state_gradient_face = zeros(Float64, num_state_gradient, Nz+1)
        for il = 1:Nl
            # single colume treatment  
            ##########
            #  reconstruct interface states by looping each cell, construct the bottom and top states
            id_col = ix:Nx:Nz*Nx
            Δzc_col = @view mesh.Δzc[il, ix, :]
            state_gradient_col = @view state_gradient[il, : , id_col]
        
            for iz = 1:Nz+1
                # bottom 
                if iz == 1
                    if bc_bottom_type == "periodic"
                        
                        Δz⁻, Δz⁺ = Δzc_col[Nz], Δzc_col[1] 
                        # interpolation to get face value at izth face, and 
                        state_gradient_face[:,iz] .= (state_gradient_col[:,mod1(iz-1, Nz)]*Δz⁺ + state_gradient_col[:,mod1(iz, Nz)]*Δz⁻)/(Δz⁻ + Δz⁺)
                    elseif bc_bottom_type == "no-penetration"
                        # do nothing
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " for second order equations has not implemented")   
                    end

                    # top
                elseif iz == Nz+1
                    if bc_top_type == "periodic"
                        
                        Δz⁻, Δz⁺ = Δzc_col[Nz], Δzc_col[1] 
                        # interpolation to get face value at izth face, and 
                        state_gradient_face[:,iz] .= (state_gradient_col[:,mod1(iz-1, Nz)]*Δz⁺ + state_gradient_col[:,mod1(iz, Nz)]*Δz⁻)/(Δz⁻ + Δz⁺)
                    elseif bc_top_type == "no-penetration"
                        # do nothing
                    else
                        error("bc_top_type = ", bc_top_type, " for second order equations has not implemented")   
                    end

                else
                    
                    Δz⁻, Δz⁺ = Δzc_col[iz-1], Δzc_col[iz] 
                    # interpolation to get face value at izth face, and 
                    state_gradient_face[:,iz] .= (state_gradient_col[:,mod1(iz-1, Nz)]*Δz⁺ + state_gradient_col[:,mod1(iz, Nz)]*Δz⁻)/(Δz⁻ + Δz⁺)
                end

                
            end
            
            
            ##########################################################################################################
            # compute vertical gradient ∂Y/∂η flux 
            
            
            # loop each element 
            for iz = 1:Nz
                
                e = ix + (iz-1)*Nx
                
                # bottom 
                if iz == 1
                    if bc_bottom_type == "periodic"
                        # use the auxiliary state in  local_aux⁺
                        ∇ref_state_gradient[il, :, e, 2] = (state_gradient_face[:,iz+1] - state_gradient_face[:,iz])/2.0
                    elseif bc_bottom_type == "no-penetration"
                        # todo one side gradient
                        ∇ref_state_gradient[il, :, e, 2] = (state_gradient_face[:,iz+1] - state_gradient_col[:,iz])
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " for second order equations has not implemented")   
                    end

                    
                    # top 
                elseif iz == Nz
                    if bc_top_type == "periodic"
                        # use the auxiliary state in  local_aux⁻
                        ∇ref_state_gradient[il, :, e, 2] = (state_gradient_face[:,iz+1] - state_gradient_face[:,iz])/2.0
                    elseif bc_top_type == "no-penetration"
                        # todo one side gradient
                        ∇ref_state_gradient[il, :, e, 2] = (state_gradient_col[:, iz] - state_gradient_col[:,iz])   
                    else
                        error("bc_bottom_type = ", bc_bottom_type, " for second order equations has not implemented")   
                    end
      
                else
                    # todo we might need limiter there
                    
                    
                    ∇ref_state_gradient[il, :, e, 2] = (state_gradient_face[:,iz+1] - state_gradient_face[:,iz])/2.0
                    
                end
            end 
            
        end
    end
end