include("TemperatureProfiles.jl")
include("TurbulenceClosures.jl")

abstract type TurbulenceClosure end


mutable struct DryAtmo <: Application
    num_state_prognostic::Int64
    num_state_diagnostic::Int64
    num_state_auxiliary::Int64
    
    # 0 means Euler 
    num_state_gradient::Int64
    
    
    bc_bottom_type::String
    bc_bottom_data::Union{Array{Float64, 1}, Nothing}
    
    bc_top_type::String
    bc_top_data::Union{Array{Float64, 1}, Nothing}
    
    bc_left_type::String
    bc_left_data::Union{Array{Float64, 1}, Nothing}
    
    bc_right_type::String
    bc_right_data::Union{Array{Float64, 1}, Nothing}
    
    hydrostatic_balance::Bool
    
    # constant diffusivity 
    diffusion_model::TurbulenceClosure
    ν::Float64
    Pr::Float64
    Δₕ::Float64
    Δᵥ::Float64
    
    g::Float64
    γ::Float64
    Rd::Float64
    MSLP::Float64
    
    
    Δt::Float64
    zT::Float64
    zD::Float64
    xT::Float64
    xD::Float64
    u_sponge::Array{Float64,1}
    
end

# ======== Turbulence Closure Content ============= # 

# Fallback method
function compute_diffusivity(
    ::TurbulenceClosure, 
    app::DryAtmo,
    ∇u::Array{Float64,2}
  )
  nothing
  @warn ("Default values from `app` are being used since no TurbulenceClosure is specified.") maxlog = 1
  return (app.ν, app.ν / app.Pr)
end

#
### Smagorinsky
#

struct Smagorinsky <: TurbulenceClosure 
  C_s::Float64
end

function compute_diffusivity(
    m::Smagorinsky, 
    app::DryAtmo,
    ∇u::Array{Float64,2},
  )
  C_s = m.C_s
  Δ = app.Δₕ
  S⃗ = (∇u + ∇u')/2
  ν = (C_s * Δ)^2 * sqrt(2 * sum(S⃗ * S⃗))
  D = ν ./ app.Pr 
  return (ν,D)
end

#
### Constant Kinematic Viscosity [m^2/s]
#
struct ConstantKinematic <: TurbulenceClosure
  ν::Float64
end
function compute_diffusivity(
    m::ConstantKinematic,
    app::DryAtmo,
    ∇u::Array{Float64,2},
  )
  return (m.ν, m.ν/ app.Pr)
end

# ======== Turbulence Closure Content ============= # 


function DryAtmo(bc_bottom_type::String,  bc_bottom_data::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String,     bc_top_data::Union{Array{Float64, 1}, Nothing},
    bc_left_type::String,    bc_left_data::Union{Array{Float64, 1}, Nothing},
    bc_right_type::String,   bc_right_data::Union{Array{Float64, 1}, Nothing},
    viscous::Bool, 
    diffusion_model::TurbulenceClosure,
    ν::Float64,
    Δₕ::Float64, Δᵥ::Float64,
    Pr::Float64, 
    gravity::Bool, hydrostatic_balance::Bool)
    
    num_state_prognostic = 4
    num_state_diagnostic = 4
    # Φ ∇Φ ρ_ref p_ref
    num_state_auxiliary = 5
    
    # u, v, T
    num_state_gradient = (viscous ? 3 : 0)

    
    
    # constant
    if gravity == false
        g = 0.0
        @assert(hydrostatic_balance == false)
        hydrostatic_balance = false
    else
        g = 9.8
    end
    
    γ = 1.4
    Rd = 287.058
    MSLP = 1.01325e5
    
    Δt, zT, zD, xT, xD, u_sponge = -1.0, -1.0, Inf64, -1.0, Inf64, [0.0, 0.0]
    
    
    DryAtmo(num_state_prognostic, num_state_diagnostic, num_state_auxiliary, num_state_gradient,
    bc_bottom_type, bc_bottom_data,
    bc_top_type, bc_top_data,
    bc_left_type, bc_left_data,
    bc_right_type, bc_right_data,
    hydrostatic_balance,
    diffusion_model,
    ν, Pr, Δₕ, Δᵥ,
    g, γ, Rd, MSLP,
    Δt, zT, zD, 
    xT, xD, u_sponge)
    
end



function update_sponge_params!(app::DryAtmo, Δt::Float64=app.Δt, zT::Float64=app.zT, zD::Float64=app.zD, 
    xT::Float64=app.xT, xD::Float64=app.xD, u_sponge::Array{Float64,1}=app.u_sponge)
    app.Δt, app.zT, app.zD, app.u_sponge = Δt, zT, zD, u_sponge
    app.xT, app.xD = xT, xD
end



function compute_min_max(app::DryAtmo, state_primitive::Array{Float64,3})
    
    @info "min ρ = ", minimum(state_primitive[:,1,:]), " max ρ = ", maximum(state_primitive[:,1,:])
    @info "min u = ", minimum(state_primitive[:,2,:]), " max u = ", maximum(state_primitive[:,2,:])
    @info "min w = ", minimum(state_primitive[:,3,:]), " max w = ", maximum(state_primitive[:,3,:])
    @info "min p = ", minimum(state_primitive[:,4,:]), " max p = ", maximum(state_primitive[:,4,:])
    
end

function internal_energy(app::DryAtmo, ρ::Float64, ρu::Array{Float64,1}, ρe::Float64, Φ::Float64)
    ρinv = 1 / ρ
    ρe_int = ρe - ρinv * sum(abs2, ρu) / 2 - ρ*Φ
    e_int = ρinv * ρe_int
    return e_int
end

# e_int = CᵥT = p/(ρ(γ-1))
function air_pressure(app::DryAtmo, ρ::Float64,  e_int::Float64)
    γ = app.γ
    p = e_int*ρ*(γ - 1)
    return p
    
end

function soundspeed_air(app::DryAtmo, ρ::Float64,  p::Float64)

    γ = app.γ
    return sqrt(γ * p / ρ)
end

function total_specific_enthalpy(app::DryAtmo, ρe::Float64, ρ::Float64, p::Float64)
    return (ρe + p)/ρ
end

function compute_wave_speed(app::DryAtmo, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
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


function prog_to_prim(app::DryAtmo, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    dim = 2
    
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    return [ρ; u; p]
end

function prim_to_prog(app::DryAtmo, state_primitive::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    dim = 2
    γ = app.γ
    ρ, u, p = state_primitive[1], state_primitive[2:dim+1], state_primitive[dim+2]
    Φ = state_auxiliary[1]
    ρu = ρ*u
    
    ρe = p/(γ-1) + 0.5*(ρu[1]*u[1] + ρu[2]*u[2]) + ρ*Φ
    
    return [ρ; ρu; ρe]
end


function prog_to_prim!(app::DryAtmo, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3}, state_primitive::Array{Float64, 3})
    # state_primitive = size(Nl, num_state_prognostic, Nz+1)
    for il = 1:size(state_prognostic, 1)
        for e = 1:size(state_prognostic, 3)
            state_primitive[il, :, e] .= prog_to_prim(app, state_prognostic[il, :, e], state_auxiliary[il, :, e])
        end
    end
end



function compute_gradient_variables!(app::DryAtmo, state_prognostic::Array{Float64,3}, state_primitive::Array{Float64,3}, 
    state_auxiliary_vol_l::Array{Float64,3}, state_gradient::Array{Float64, 3})

    for il = 1:size(state_gradient, 1)
        for e = 1:size(state_gradient, 3)
            ρ, u, v, p = state_primitive[il, :, e] 
            ρe = state_prognostic[il, 4, e]
            h = (ρe + p)/ρ
            state_gradient[il, :, e] .= u, v, h
        end
    end

end



function flux_first_order(app::DryAtmo, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    
    dim = 2
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    p_ref = state_auxiliary[4]
    
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    
    return flux_first_order(app, ρ, ρu, ρe, p, p_ref)
end


function flux_first_order(app::DryAtmo, ρ::Float64, ρu::Array{Float64,1}, ρe::Float64, p::Float64, p_ref::Float64)
    
    # test
    # p_ref = 0.0
    
    u = ρu/ρ
    flux = 
    [ρu[1] ρu[2];
    ρu[1]*u[1]+p-p_ref   ρu[1]*u[2];
    ρu[2]*u[1]     ρu[2]*u[2]+p-p_ref; 
    (ρe+p)*u[1]    (ρe+p)*u[2]]
    
    
    # @info p, p_ref
    
    
    return flux
end


# Roe flux
roe_average(sqrt_ρ, var⁻, var⁺) =
(var⁻ .+ sqrt_ρ * var⁺) / (1.0 + sqrt_ρ)

function numerical_flux_first_order(app::DryAtmo, 
    state_prognostic⁻::Array{Float64, 1}, state_auxiliary⁻::Array{Float64, 1},
    state_prognostic⁺::Array{Float64, 1}, state_auxiliary⁺::Array{Float64, 1},
    n::Array{Float64, 1})
    
    dim = 2
    
    n_len = sqrt(n[1]^2 + n[2]^2)
    n_ij = n / n_len
    t_ij = [-n_ij[2], n_ij[1]]
    
    Φ = state_auxiliary⁻[1]
    p_ref⁻, p_ref⁺ = state_auxiliary⁻[4], state_auxiliary⁺[4]
    
    
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
    
    flux⁻ = flux_first_order(app, ρ⁻, ρu⁻, ρe⁻, p⁻, p_ref⁻) * n_ij
    flux⁺ = flux_first_order(app, ρ⁺, ρu⁺, ρe⁺, p⁺, p_ref⁺) * n_ij
    
    
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
    
    return n_len*flux
end



function flux_second_order(app::DryAtmo, state_prognostic::Array{Float64, 1}, ∇state_gradient::Array{Float64, 2}, state_auxiliary::Array{Float64, 1})
    # Update ν, D_t using dynamic model
    ∇u, ∇h =  ∇state_gradient[1:2, :], ∇state_gradient[3, :]
    # Compute strain-rate tensor (symmetric)
    ν, D_t = compute_diffusivity(app.diffusion_model, app, ∇u)
    S⃗ = (∇u + ∇u')/2
    τ = -2*ν*S⃗
    ρ, ρu = state_prognostic[1], state_prognostic[2:3]
    u = ρu/ρ
    # Return second order fluxes
    return  [0.0 0.0;
            ρ*τ;
            (τ * ρ*u - ρ*D_t*∇h)']
    
end
function flux_second_order_prim(app::DryAtmo, state_primitive::Array{Float64, 1}, ∇state_gradient::Array{Float64, 2}, state_auxiliary::Array{Float64, 1})
    # Update ν, D_t using dynamic model
    ∇u, ∇h =  ∇state_gradient[1:2, :], ∇state_gradient[3, :]
    # Compute strain-rate tensor (symmetric)
    ν, D_t=  compute_diffusivity(app.diffusion_model, app,∇u)
    S⃗ = (∇u + ∇u')/2
    τ = -2*ν*S⃗
    ρ, u = state_primitive[1], state_primitive[2:3]
    # Return second order fluxes
    return  [0.0 0.0;
            ρ*τ;
            (τ * ρ*u - ρ*D_t*∇h)']
end       


# Central flux
function numerical_flux_second_order(app::DryAtmo, state_prognostic⁻::Array{Float64, 1}, ∇state_gradient⁻::Array{Float64, 2}, state_auxiliary⁻::Array{Float64, 1}, 
    state_prognostic⁺::Array{Float64, 1}, ∇state_gradient⁺::Array{Float64, 2}, state_auxiliary⁺::Array{Float64, 1}, 
    n::Array{Float64, 1})

    flux⁻ = flux_second_order(app, state_prognostic⁻, ∇state_gradient⁻, state_auxiliary⁻)
    flux⁺ = flux_second_order(app, state_prognostic⁺, ∇state_gradient⁺, state_auxiliary⁺)
    return  0.5*(flux⁻ + flux⁺) * n
    
end

function bc_second_order_flux(app::DryAtmo, state_primitive::Array{Float64, 1}, bc_type::String, bc_data::Array{Float64, 1})
    @assert(bc_type == "no-penetration")
    ρ = state_primitive[1]
    # τn = -0.5*ν*(∇u + ∇u')n, JD = - ν/Pr*∇h n
    τn, JD = bc_data[1:2], bc_data[3]
    return [0.0; ρ*τn; ρ*JD]
end


# Wall Flux
# Primitive state variable vector V_i
# outward wall normal n_i
function wall_flux_first_order(app::DryAtmo, 
    state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1}, 
    n::Array{Float64, 1})
    
    dim = 2
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    p_ref = state_auxiliary[4]
    
    return [0.0, (p - p_ref) * n[1] , (p - p_ref) * n[2], 0.0]
end


function source(app::DryAtmo, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1}, x::Float64)
    ρ = state_prognostic[1]
    ρ_ref = state_auxiliary[5]
    ∇Φ = state_auxiliary[2:3]
    source = [0.0; -(ρ - ρ_ref)*∇Φ ; 0.0]
    
    
    # sponge layer
    g = app.g
    z = state_auxiliary[1]/g
    Δt, zT, zD, xT, xD = app.Δt, app.zT, app.zD, app.xT, app.xD
    ρ, ρu = state_prognostic[1], state_prognostic[2:3]
    α = 1.0/(20Δt)
    β = 0.0
    if z > zD 
        β =  α*sin(π/2.0 * ((z - zD)/(zT - zD)))^2
    end
    if abs(x) > xD
        β = max(β, α*sin(π/2.0 * ((abs(x) - xD)/(xT - xD)))^2)
    end
    source[2:3] .-= β*(ρu - ρ*app.u_sponge)
    
    return source
end

# The auxiliary state has Φ and ∇Φ
function init_state_auxiliary!(app::DryAtmo, mesh::Mesh, 
    state_auxiliary_vol_l::Array{Float64, 3}, state_auxiliary_vol_q::Array{Float64, 3}, 
    state_auxiliary_surf_h::Array{Float64, 4}, state_auxiliary_surf_v::Array{Float64, 4})
    vol_l_geo, vol_q_geo, sgeo_h, sgeo_v = mesh.vol_l_geo, mesh.vol_q_geo, mesh.sgeo_h, mesh.sgeo_v
    g = app.g
    topology_type = mesh.topology_type
    ϕl_q = mesh.ϕl_q
    
    
    if topology_type == "AtmoLES"
        
        x, z = vol_l_geo[1, :, :], vol_l_geo[2, :, :]
        state_auxiliary_vol_l[:, 1, :] .= g*z
        state_auxiliary_vol_l[:, 2, :] .= 0.0
        state_auxiliary_vol_l[:, 3, :] .= g
        
        
        state_auxiliary_vol_q[:, 1, :] .= ϕl_q * state_auxiliary_vol_l[:, 1, :]
        state_auxiliary_vol_q[:, 2, :] .= ϕl_q * state_auxiliary_vol_l[:, 2, :]
        state_auxiliary_vol_q[:, 3, :] .= ϕl_q * state_auxiliary_vol_l[:, 3, :]
        
        x, z = sgeo_h[4, :, :, :], sgeo_h[5, :, :, :] 
        state_auxiliary_surf_h[:, 1, :, :] .= g*z
        state_auxiliary_surf_h[:, 2, :, :] .= 0.0
        state_auxiliary_surf_h[:, 3, :, :] .= g
        
        x, z = sgeo_v[4, :, :, :], sgeo_v[5, :, :, :] 
        state_auxiliary_surf_v[:, 1, :, :] .= g*z
        state_auxiliary_surf_v[:, 2, :, :] .= 0.0
        state_auxiliary_surf_v[:, 3, :, :] .= g
        
    elseif topology_type == "AtmoGCM"
        # The center is at [0, 0]
        # For ∇Φ, we use ∇Φ = g∇alt = g*n_surf_v, instead of 
        x, z = vol_l_geo[1, :, :], vol_l_geo[2, :, :]
        r = sqrt.(x.^2 + z.^2)
        
        k1 = sgeo_v[1, :, 2, :]./sqrt.(sgeo_v[1, :, 2, :].^2 + sgeo_v[2, :, 2, :].^2) 
        k2 = sgeo_v[2, :, 2, :]./sqrt.(sgeo_v[1, :, 2, :].^2 + sgeo_v[2, :, 2, :].^2) 
        
        state_auxiliary_vol_l[:, 1, :] .= g * (r .- mesh.topology_size[1])
        state_auxiliary_vol_l[:, 2, :] .= g * k1
        state_auxiliary_vol_l[:, 3, :] .= g * k2
        
        x, z = vol_q_geo[6, :, :], vol_q_geo[7, :, :]
        r = sqrt.(x.^2 + z.^2)
        state_auxiliary_vol_q[:, 1, :] .= ϕl_q * state_auxiliary_vol_l[:, 1, :]
        state_auxiliary_vol_q[:, 2, :] .= ϕl_q * state_auxiliary_vol_l[:, 2, :]
        state_auxiliary_vol_q[:, 3, :] .= ϕl_q * state_auxiliary_vol_l[:, 3, :]
        
        x, z = sgeo_h[4, :, :, :], sgeo_h[5, :, :, :] 
        r = sqrt.(x.^2 + z.^2)
        state_auxiliary_surf_h[:, 1, :, :] .= g * (r .- mesh.topology_size[1])
        state_auxiliary_surf_h[:, 2, :, :] .= NaN64
        state_auxiliary_surf_h[:, 3, :, :] .= NaN64
        
        x, z = sgeo_v[4, :, :, :], sgeo_v[5, :, :, :] 
        r = sqrt.(x.^2 + z.^2)
        state_auxiliary_surf_v[:, 1, :, :] .= g * (r .- mesh.topology_size[1])
        state_auxiliary_surf_v[:, 2, :, :] .= NaN64
        state_auxiliary_surf_v[:, 3, :, :] .= NaN64
        
    else
        error("topology_type : ", topology_type, " has not implemented")
    end
end





#################################################################################################
"""
update state_prognostic
and app.bc_top_data
"""
function init_discrete_hydrostatic_balance!(app::DryAtmo, mesh::Mesh, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3},
    T_virt_surf::Float64, T_min_ref::Float64, H_t::Float64)
    
    
    Nl,Nx,Nz = mesh.Nl, mesh.Nx, mesh.Nz
    
    profile = DecayingTemperatureProfile(app, T_virt_surf, T_min_ref, H_t)
    γ = app.γ
    
    topology_type = mesh.topology_type
    topology_size = mesh.topology_size
    
    vol_l_geo = mesh.vol_l_geo
    u_init = [0.0; 0.0]
    Δzc = mesh.Δzc
    g = app.g
    @assert(g > 0.0)
    
    for ix = 1:Nx
        for il = 1:Nl
            
            p⁻, ρ⁻, Δz⁻ = 0.0, 0.0, 0.0
            for iz = 1:Nz
                e  = ix + (iz - 1)*Nx
                # p_{iz - 1}  - p_{iz} - = ρ_{iz}*g*Δz_{iz}/2 + ρ_{iz-1}*g*Δz_{iz-1}/2
                Δz = Δzc[il, ix, iz]
                Φ = state_auxiliary[il, 1, e]
                alt = Φ/g
                Tv, p, ρ = profile(alt)
                
                if iz > 1
                    # hydrostatic correction
                    # @show ρ, (p⁻ - p -  ρ⁻*g*Δz⁻/2.0)/ (g*Δz/2.0)
                    ρ = (p⁻ - p -  ρ⁻*g*Δz⁻/2.0)/ (g*Δz/2.0)
                    
                end
                p⁻, ρ⁻, Δz⁻  = p, ρ, Δz
                
                # if ix == 1 && il == 1
                #     @show iz, p, ρ, Δz
                # end
                
                
                ρu = ρ*u_init
                ρe = p/(γ-1) + 0.5*(ρu[1]*u_init[1] + ρu[2]*u_init[2]) + ρ*Φ
                
                state_prognostic[il, :, e] .= [ρ ; ρu ; ρe]
                
                # @show ρ, u_init, p
                # @show prog_to_prim(app, state_prognostic[il, :, e], state_auxiliary[il, :, e])
                
                # error("stop")
            end
            
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


#################################################################################################
"""
update state_prognostic
and app.bc_top_data
"""
#################################################################################################
"""
update state_prognostic
and app.bc_top_data
"""
function init_discrete_hydrostatic_balance!(app::DryAtmo, mesh::Mesh, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3},
    f_profile::Function, u_init::Array{Float64, 1})
    
    @show  f_profile
    
    Nl,Nx,Nz = mesh.Nl, mesh.Nx, mesh.Nz
    
    
    γ = app.γ
    
    topology_type = mesh.topology_type
    topology_size = mesh.topology_size
    
    vol_l_geo = mesh.vol_l_geo
    
    Δzc = mesh.Δzc
    g = app.g
    
    
    for ix = 1:Nx
        for il = 1:Nl
            
            p⁻, ρ⁻, Δz⁻ = 0.0, 0.0, 0.0
            for iz = 1:Nz
                e  = ix + (iz - 1)*Nx
                # p_{iz - 1}  - p_{iz} - = ρ_{iz}*g*Δz_{iz}/2 + ρ_{iz-1}*g*Δz_{iz-1}/2
                Δz = Δzc[il, ix, iz]
                Φ = state_auxiliary[il, 1, e]
                alt = Φ/g
                p, ρ = f_profile(alt)
                ρ_o = ρ
                if iz > 1
                    # hydrostatic correction
                    # @show ρ, (p⁻ - p -  ρ⁻*g*Δz⁻/2.0)/ (g*Δz/2.0)
                    ρ = (p⁻ - p -  ρ⁻*g*Δz⁻/2.0)/ (g*Δz/2.0)
                    
                end
                p⁻, ρ⁻, Δz⁻  = p, ρ, Δz
                
                
                
                ρu = ρ*u_init
                ρe = p/(γ-1) + 0.5*(ρu[1]*u_init[1] + ρu[2]*u_init[2]) + ρ*Φ
                
                # @info ρ, ρ_o, p
                state_prognostic[il, :, e] .= [ρ ; ρu ; ρe]
                
            end
        end
    end
    
    return 
end

# initialize 
function init_state!(app::DryAtmo, mesh::Mesh, state_prognostic::Array{Float64, 3}, func::Function)
    
    Nl, num_state_prognostic, nelem = size(state_prognostic)
    vol_l_geo = mesh.vol_l_geo
    
    for e = 1:nelem
        for il = 1:Nl
            
            x, z = vol_l_geo[1:2, il, e]
            
            state_prognostic[il, :, e] .= func(x, z)
        end
    end
end


function bc_impose(app::DryAtmo, state_primitive::Array{Float64, 1}, bc_type::String, n::Array{Float64, 1})
    
    ρ, u, p = state_primitive[1], state_primitive[2:3], state_primitive[4]
    if bc_type == "no-slip"
        u_g = [0.0 ;0.0]
    elseif bc_type == "no-penetration"
        n_len_2 = n[1]^2 + n[2]^2
        u_g = u - (u * n')*n/n_len_2 
    else
        error("bc_type  : ", bc_type )
    end
    return [ρ ; u_g ; p ]
end



function DryAtmo_test()
    app = DryAtmo("no-slip", "no-slip")
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
