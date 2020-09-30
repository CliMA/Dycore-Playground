include("TemperatureProfiles.jl")

mutable struct DryEuler <: Application
    num_state_prognostic::Int64
    num_state_diagnostic::Int64
    num_state_auxiliary::Int64
    
    bc_bottom_type::String
    bc_bottom_data::Union{Array{Float64, 1}, Nothing}
    
    bc_top_type::String
    bc_top_data::Union{Array{Float64, 1}, Nothing}
    
    bc_left_type::String
    bc_left_data::Union{Array{Float64, 1}, Nothing}
    
    bc_right_type::String
    bc_right_data::Union{Array{Float64, 1}, Nothing}
    
    
    
    g::Float64
    γ::Float64
    Rd::Float64
    MSLP::Float64
end

function DryEuler(bc_bottom_type::String,  bc_bottom_data::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String,     bc_top_data::Union{Array{Float64, 1}, Nothing},
    bc_left_type::String,    bc_left_data::Union{Array{Float64, 1}, Nothing},
    bc_right_type::String,   bc_right_data::Union{Array{Float64, 1}, Nothing},
    gravity::Bool)
    
    num_state_prognostic = 4
    num_state_diagnostic = 4
    num_state_auxiliary = 3
    
    # constant
    if gravity == false
        g = 0.0
    else
        g = 9.8
    end
    
    γ = 1.4
    Rd = 287.058
    MSLP = 1.01325e5
    
    DryEuler(num_state_prognostic, num_state_diagnostic, num_state_auxiliary,
    bc_bottom_type, bc_bottom_data,
    bc_top_type, bc_top_data,
    bc_left_type, bc_left_data,
    bc_right_type, bc_right_data,
    g, γ, Rd, MSLP)
end


function internal_energy(app::DryEuler, ρ::Float64, ρu::Array{Float64,1}, ρe::Float64, Φ::Float64)
    ρinv = 1 / ρ
    ρe_int = ρe - ρinv * sum(abs2, ρu) / 2 - ρ*Φ
    e_int = ρinv * ρe_int
    return e_int
end

# e_int = CᵥT = p/(ρ(γ-1))
function air_pressure(app::DryEuler, ρ::Float64,  e_int::Float64)
    γ = app.γ
    p = e_int*ρ*(γ - 1)
    return p
    
end

function soundspeed_air(app::DryEuler, ρ::Float64,  p::Float64)
    γ = app.γ
    return sqrt(γ * p / ρ)
end

function total_specific_enthalpy(app::DryEuler, ρe::Float64, ρ::Float64, p::Float64)
    return (ρe + p)/ρ
end

function compute_wave_speed(app::DryEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    # c + |u|
    dim = 2
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    c = soundspeed_air(app, ρ,  p)
    
    return c + sqrt(u[1]^2 + u[2]^2)
    
    
end


function prog_to_prim(app::DryEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    dim = 2
    
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    return [ρ; u; p]
end

function prim_to_prog(app::DryEuler, state_primitive::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    dim = 2
    γ = app.γ
    ρ, u, p = state_primitive[1], state_primitive[2:dim+1], state_primitive[dim+2]
    Φ = state_auxiliary[1]
    ρu = ρ*u
    
    ρe = p/(γ-1) + 0.5*(ρu[1]*u[1] + ρu[2]*u[2]) + ρ*Φ
    
    return [ρ; ρu; ρe]
end


# function prim_to_prog!(app::DryEuler, state_primitive::Array{Float64, 2}, state_auxiliary::Array{Float64, 2}, state_prognostic::Array{Float64, 2})
#     # state_primitive = size(num_state_prognostic, Nz+1)
#     for iz = 1:size(state_primitive, 2)
#         state_prognostic[:, iz] .= prim_to_prog(app, state_primitive[:, iz], state_auxiliary[:, iz])
#     end
# end

function prog_to_prim!(app::DryEuler, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3}, state_primitive::Array{Float64, 3})
    # state_primitive = size(Nl, num_state_prognostic, Nz+1)
    for il = 1:size(state_prognostic, 1)
        for iz = 1:size(state_prognostic, 3)
            state_primitive[il, :, iz] .= prog_to_prim(app, state_prognostic[il, :, iz], state_auxiliary[il, :, iz])
        end
    end
end

function flux_first_order(app::DryEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    
    dim = 2
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    return flux_first_order(app, ρ, ρu, ρe, p)
end


function flux_first_order(app::DryEuler, ρ::Float64, ρu::Array{Float64,1}, ρe::Float64, p::Float64)
    
    
    u = ρu/ρ
    flux = 
    [ρu[1] ρu[2];
    ρu[1]*u[1]+p   ρu[1]*u[2];
    ρu[2]*u[1]     ρu[2]*u[2]+p; 
    (ρe+p)*u[1]    (ρe+p)*u[2]]
    
    return flux
end


# Roe flux
roe_average(sqrt_ρ, var⁻, var⁺) =
(var⁻ .+ sqrt_ρ * var⁺) / (1.0 + sqrt_ρ)

function numerical_flux_first_order(app::DryEuler, 
    state_prognostic⁻::Array{Float64, 1}, state_auxiliary⁻::Array{Float64, 1},
    state_prognostic⁺::Array{Float64, 1}, state_auxiliary⁺::Array{Float64, 1},
    n::Array{Float64, 1})
    
    dim = 2
    
    n_len = sqrt(n[1]^2 + n[2]^2)
    n_ij = n / n_len
    t_ij = [-n_ij[2], n_ij[1]]
    
    Φ = state_auxiliary⁻[1]
    γ = app.γ
    
    ρ⁻, ρu⁻, ρe⁻ = state_prognostic⁻[1], state_prognostic⁻[2:dim+1], state_prognostic⁻[dim+2]
    u⁻ = ρu⁻/ρ⁻
    e_int⁻ = internal_energy(app, ρ⁻, ρu⁻, ρe⁻, Φ)
    p⁻ = air_pressure(app, ρ⁻,  e_int⁻)
    a⁻ = soundspeed_air(app, ρ⁻,  p⁻)
    h⁻ = total_specific_enthalpy(app, ρe⁻, ρ⁻,  p⁻)
    
    ρ⁺, ρu⁺, ρe⁺ = state_prognostic⁺[1], state_prognostic⁺[2:dim+1], state_prognostic⁺[dim+2]
    u⁺ = ρu⁺/ρ⁺
    e_int⁺ = internal_energy(app, ρ⁺, ρu⁺, ρe⁺, Φ)
    p⁺ = air_pressure(app, ρ⁺,  e_int⁺)
    a⁺ = soundspeed_air(app, ρ⁺,  p⁺)
    h⁺ = total_specific_enthalpy(app, ρe⁺, ρ⁺,  p⁺)
    
    flux⁻ = flux_first_order(app, ρ⁻, ρu⁻, ρe⁻, p⁻) * n_ij
    flux⁺ = flux_first_order(app, ρ⁺, ρu⁺, ρe⁺, p⁺) * n_ij
    
    un⁻= u⁻' * n_ij
    ut⁻= u⁻' * t_ij
    
    # right state
    
    un⁺ = u⁺' * n_ij
    ut⁺ = u⁺' * t_ij
    
    # compute the Roe-averaged quatities
    sqrt_ρ = sqrt(ρ⁺ / ρ⁻)
    ρ_rl = sqrt_ρ * ρ⁻
    u_rl = roe_average(sqrt_ρ, u⁻, u⁺)
    h_rl = roe_average(sqrt_ρ, h⁻, h⁺)
    un_rl = roe_average(sqrt_ρ, un⁻, un⁺)
    ut_rl = roe_average(sqrt_ρ, ut⁻, ut⁺)
    
    # Todo this is different with CliMA
    a_rl = sqrt((γ - 1) * (h_rl - 0.5 * (u_rl[1]^2 + u_rl[2]^2 ) - Φ))
    
    # wave strengths
    dp = p⁺ - p⁻
    dρ = ρ⁺ - ρ⁻
    dun = un⁺ - un⁻
    dut = ut⁺ - ut⁻
    du = [(dp - ρ_rl * a_rl * dun) / (2.0 * a_rl * a_rl),  ρ_rl * dut,  dρ - dp / (a_rl^2),  (dp + ρ_rl * a_rl * dun) / (2.0 * a_rl^2)]
    
    # compute the Roe-average wave speeds
    ws = [abs(un_rl - a_rl), abs(un_rl), abs(un_rl), abs(un_rl + a_rl)]
    
    
    # compute the right characteristic eigenvectors
    P_inv = [[1.0                    0.0          1.0                            1.0];
    [u_rl[1] - a_rl * n_ij[1]    t_ij[1]        u_rl[1]                           u_rl[1] + a_rl * n_ij[1]];
    [u_rl[2] - a_rl * n_ij[2]    t_ij[2]        u_rl[2]                           u_rl[2] + a_rl * n_ij[2]];
    [h_rl - un_rl * a_rl       ut_rl            0.5 * (u_rl[1] * u_rl[1] + u_rl[2] * u_rl[2])      h_rl + un_rl * a_rl]]
    
    
    
    flux = 0.5 * (flux⁺ + flux⁻ - P_inv * (du .* ws))
    

    @info " dp : ", dp, flux, n_len, n_len*flux
    return n_len*flux
end


# Wall Flux
# Primitive state variable vector V_i
# outward wall normal n_i
function wall_flux_first_order(app::DryEuler, 
    state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1}, 
    n::Array{Float64, 1})
    
    dim = 2
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    return [0.0, p * n[1] , p * n[2], 0.0]
end


function source(app::DryEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    ρ = state_prognostic[1]
    ∇Φ = state_auxiliary[2:3]
    return [0.0; -ρ*∇Φ; 0.0]
end


# The auxiliary state has Φ and ∇Φ
function init_state_auxiliary!(app::DryEuler, mesh::Mesh, 
    state_auxiliary_vol_l::Array{Float64, 3}, state_auxiliary_vol_q::Array{Float64, 3}, 
    state_auxiliary_surf_h::Array{Float64, 4}, state_auxiliary_surf_v::Array{Float64, 4})
    
    vol_l_geo, vol_q_geo, sgeo_h, sgeo_v = mesh.vol_l_geo, mesh.vol_q_geo, mesh.sgeo_h, mesh.sgeo_v
    
    g = app.g
    topology_type = mesh.topology_type
    
    if topology_type == "AtmoLES"
        
        x, z = vol_l_geo[1, :, :], vol_l_geo[2, :, :]
        state_auxiliary_vol_l[:, 1, :]  = g*z
        state_auxiliary_vol_l[:, 2, :] .= 0.0
        state_auxiliary_vol_l[:, 3, :] .= g
        
        x, z = vol_q_geo[6, :, :], vol_q_geo[7, :, :]
        state_auxiliary_vol_q[:, 1, :]  = g*z
        state_auxiliary_vol_q[:, 2, :] .= 0.0
        state_auxiliary_vol_q[:, 3, :] .= g
        
        x, z = sgeo_h[4, :, :, :], sgeo_h[5, :, :, :] 
        state_auxiliary_surf_h[:, 1, :, :]  = g*z
        state_auxiliary_surf_h[:, 2, :, :] .= 0.0
        state_auxiliary_surf_h[:, 3, :, :] .= g
        
        x, z = sgeo_v[4, :, :, :], sgeo_v[5, :, :, :] 
        state_auxiliary_surf_v[:, 1, :, :]  = g*z
        state_auxiliary_surf_v[:, 2, :, :] .= 0.0
        state_auxiliary_surf_v[:, 3, :, :] .= g
        
    elseif topology_type == "AtmoGCM"
        # The center is at [0, 0]
        
        x, z = vol_l_geo[1, :, :], vol_l_geo[2, :, :]
        r = sqrt(x.^2 + z.^2)
        state_auxiliary_vol_l[:, 1, :] .= g * (r - mesh.topology_size[1])
        state_auxiliary_vol_l[:, 2, :] .= g * x./r
        state_auxiliary_vol_l[:, 3, :] .= g * z./r
        
        x, z = vol_q_geo[6, :, :], vol_l_geo[7, :, :]
        r = sqrt(x.^2 + z.^2)
        state_auxiliary_vol_q[:, 1, :] .= g * (r - mesh.topology_size[1])
        state_auxiliary_vol_q[:, 2, :] .= g * x./r
        state_auxiliary_vol_q[:, 3, :] .= g * z./r
        
        x, z = sgeo_h[4, :, :, :], sgeo_h[5, :, :, :] 
        r = sqrt(x.^2 + z.^2)
        state_auxiliary_surf_h[:, 1, :, :] .= g * (r - mesh.topology_size[1])
        state_auxiliary_surf_h[:, 2, :, :] .= g * x./r
        state_auxiliary_surf_h[:, 3, :, :] .= g * z./r
        
        x, z = sgeo_v[4, :, :, :], sgeo_v[5, :, :, :] 
        r = sqrt(x.^2 + z.^2)
        state_auxiliary_surf_v[:, 1, :, :] .= g * (r - mesh.topology_size[1])
        state_auxiliary_surf_v[:, 2, :, :] .= g * x./r
        state_auxiliary_surf_v[:, 3, :, :] .= g * z./r
        
    else
        error("topology_type : ", topology_type, " has not implemented")
    end
end





#################################################################################################
"""
update state_prognostic
and app.bc_top_data
"""
function init_hydrostatic_balance!(app::DryEuler, mesh::Mesh, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3},
    T_virt_surf::Float64, T_min_ref::Float64, H_t::Float64)
    
    profile = DecayingTemperatureProfile(app, T_virt_surf, T_min_ref, H_t)
    γ = app.γ
    
    topology_type = mesh.topology_type
    topology_size = mesh.topology_size
    
    vol_l_geo = mesh.vol_l_geo
    u_init = [0.0; 0.0]
    
    g = app.g
    @assert(g > 0.0)
    
    for e = 1:nelem
        for il = 1:Nl
            
            Φ = state_auxiliary[il, 1, e]
            
            
            
            Tv, p, ρ = profile(Φ/g)
            # @info il, state_auxiliary[il, :, e], alt, Tv, p, ρ , Φ/g
            
            
            ρu = ρ*u_init
            ρe = p/(γ-1) + 0.5*(ρu[1]*u_init[1] + ρu[2]*u_init[2]) + ρ*Φ
            
            state_prognostic[il, :, e] .= [ρ ; ρu ; ρe]
            
            # error("stop")
        end
    end
    
    # 
    if app.bc_top_type == "outlet"
        topology = mesh.topology
        x, z = topology[:, 1, end] # anyone should be the same
        if topology_type == "AtmoLES"
            alt = z
        elseif topology_type == "AtmoGCM"
            alt = sqrt(x^2 + z^2) - topology_size[1]
        else 
            error("topology_type : ", topology_type, " has not implemented yet.")
        end
        Tv, p, ρ = profile(alt)
        app.bc_top_data = [ρ; 0.0; 0.0; p]
    end

    return profile
end

# initialize 
function init_state!(app::DryEuler, mesh::Mesh, state_prognostic::Array{Float64, 3}, func::Function)
    
    Nl, num_state_prognostic, nelem = size(state_prognostic)
    vol_l_geo = mesh.vol_l_geo
    
    for e = 1:nelem
        for il = 1:Nl
            
            x, z = vol_l_geo[1:2, il, e]
            
            state_prognostic[il, :, e] .= func(x, z)
        end
    end
end


function populate_ghost_cell(app::DryEuler, state_primitive::Array{Float64, 1}, state_primitive⁺::Array{Float64, 1}, Δz::Float64, Δz⁺::Float64, 
    bc_type::String, n::Array{Float64, 1})

    ρ, u, p = state_primitive[1], state_primitive[2:3], state_primitive[4]
    ρ⁺, p⁺ = state_primitive⁺[1], state_primitive⁺[4]
    if bc_type == "no-slip"
        u_g = -u
    elseif bc_type == "no-penetrate"
        u_g = u - 2*(u * n')*n
    else
        error("bc_type  : ", bc_type )
    end
    @info "state_primitive,, state_primitive⁺ = ",  state_primitive, state_primitive⁺
    return [ρ - 2*(ρ⁺ - ρ)/(Δz + Δz⁺) * Δz ; u_g ; p - 2*(p⁺ - p)/(Δz + Δz⁺) * Δz]
end

function DryEuler_test()
    app = DryEuler("no-slip", "no-slip")
    γ = app.γ
    ρl, ul, pl = 1.0,   [1.0;10.0], 10000.0
    # ρr, ur, pr = 0.125, [1.0;20.0], 1000.0
    ρr, ur, pr = 1.0,   [1.0;10.0], 10000.0
    
    prim_l = [ρl; ul; pl]
    prim_r = [ρr; ur; pr]
    aux = [0.0]
    
    conser_l = prim_to_prog(app, prim_l, aux)
    conser_r = prim_to_prog(app, prim_r, aux)
    
    
    prim_l_new = prog_to_prim(app, conser_l, aux)
    prim_r_new = prog_to_prim(app, conser_r, aux)
    
    @show norm(prim_l - prim_l_new)
    @show norm(prim_r - prim_r_new)
    
    
    n = [1.0;0.5]
    flux  = flux_first_order(app, conser_l, aux) * n
    roe_flux =  numerical_flux_first_order(app, conser_l, aux, conser_r, aux, n)
    
    @show norm(flux - roe_flux)
end