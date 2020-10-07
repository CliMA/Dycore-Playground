include("TemperatureProfiles.jl")

mutable struct MoistEuler <: Application
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

    use_ref_state::Bool
    
    param_set
    
    g::Float64
    γ::Float64
    Rd::Float64
    MSLP::Float64
end

function MoistEuler(bc_bottom_type::String,  bc_bottom_data::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String,     bc_top_data::Union{Array{Float64, 1}, Nothing},
    bc_left_type::String,    bc_left_data::Union{Array{Float64, 1}, Nothing},
    bc_right_type::String,   bc_right_data::Union{Array{Float64, 1}, Nothing},
    gravity::Bool)
    
    # ρ ρu ρe ρq_tot; primitive variables ρ u q q_tot
    num_state_prognostic = 5 
    num_state_diagnostic = 5
    # Φ ∇Φ ρ_ref p_ref
    num_state_auxiliary = 5
    
    # constant
    if gravity == false
        g = 0.0
        use_ref_state = false
    else
        g = 9.8
        use_ref_state = true
    end
    
    γ = 1.4
    Rd = 287.058
    MSLP = 1.01325e5
    
    MoistEuler(num_state_prognostic, num_state_diagnostic, num_state_auxiliary,
    bc_bottom_type, bc_bottom_data,
    bc_top_type, bc_top_data,
    bc_left_type, bc_left_data,
    bc_right_type, bc_right_data,
    use_ref_state,
    g, γ, Rd, MSLP)
end





function prog_to_prim(app::MoistEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    dim = 2
    
    ρ, ρu, ρe = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2]
    Φ = state_auxiliary[1]
    
    u = ρu/ρ
    e_int = internal_energy(app, ρ, ρu, ρe, Φ)
    p = air_pressure(app, ρ,  e_int)
    
    return [ρ; u; p]
end

function prim_to_prog(app::MoistEuler, state_primitive::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    dim = 2
    γ = app.γ
    ρ, u, p = state_primitive[1], state_primitive[2:dim+1], state_primitive[dim+2]
    Φ = state_auxiliary[1]
    ρu = ρ*u
    
    ρe = p/(γ-1) + 0.5*(ρu[1]*u[1] + ρu[2]*u[2]) + ρ*Φ
    
    return [ρ; ρu; ρe]
end


function prog_to_prim!(app::MoistEuler, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3}, state_primitive::Array{Float64, 3})
    # state_primitive = size(Nl, num_state_prognostic, Nz+1)
    for il = 1:size(state_prognostic, 1)
        for iz = 1:size(state_prognostic, 3)
            state_primitive[il, :, iz] .= prog_to_prim(app, state_prognostic[il, :, iz], state_auxiliary[il, :, iz])
        end
    end
end

function flux_first_order(app::MoistEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    
    dim = 2
    ρ, ρu, ρe, ρq_t = state_prognostic[1], state_prognostic[2:dim+1], state_prognostic[dim+2], state_prognostic[dim+3]
    Φ = state_auxiliary[1]

    p_ref = state_auxiliary[4]
 
    
    # ts = recover_thermo_state(m, state, aux)
    # p = air_pressure(ts)

    e_int = internal_energy(app, ρ, ρu, ρe, Φ)

    q_t = ρq_t/ρ

    PhaseEquil{eltype(state), typeof(atmos.param_set)}(
        atmos.param_set,
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        aux.moisture.temperature,
    )

    T = saturation_adjustment(
        param_set,
        e_int,
        ρ,
        q_t)




    _liquid_frac = liquid_fraction(param_set, T, phase_type)                    # fraction of condensate that is liquid
    q_c = saturation_excess(param_set, T, ρ, phase_type, PhasePartition(q_tot)) # condensate specific humidity
    q_liq = _liquid_frac * q_c                                                  # liquid specific humidity
    q_ice = (1 - _liquid_frac) * q_c                                            # ice specific humidity




        air_pressure(ts::ThermodynamicState) = air_pressure(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

    p = air_pressure(app, ρ,  T,  q)

    
    return flux_first_order(app, ρ, ρu, ρe, ρq_t, p, p_ref)
end


function flux_first_order(app::MoistEuler, ρ::Float64, ρu::Array{Float64,1}, ρe::Float64, ρq_t::Float64, p::Float64, p_ref::Float64)
    
    
    
    u = ρu/ρ
    flux = 
    [ρu[1]               ρu[2];
    ρu[1]*u[1]+p-p_ref   ρu[1]*u[2];
    ρu[2]*u[1]           ρu[2]*u[2]+p-p_ref; 
    (ρe+p)*u[1]         (ρe+p)*u[2];
    ρq_t*u[1]            ρq_t*u[2]]

    
    return flux
end


# Roe flux
roe_average(sqrt_ρ, var⁻, var⁺) =
(var⁻ .+ sqrt_ρ * var⁺) / (1.0 + sqrt_ρ)

function numerical_flux_first_order(app::MoistEuler, 
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


# Wall Flux
# Primitive state variable vector V_i
# outward wall normal n_i
function wall_flux_first_order(app::MoistEuler, 
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


function source(app::MoistEuler, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    ρ = state_prognostic[1]
    ρ_ref = state_auxiliary[5]


    ∇Φ = state_auxiliary[2:3]
    return [0.0; -(ρ - ρ_ref)*∇Φ; 0.0]
end










function bc_impose(app::MoistEuler, state_primitive::Array{Float64, 1}, bc_type::String, n::Array{Float64, 1})
    
    u = state_primitive[2:3]
    if bc_type == "no-slip"
        u_g = [0.0 ;0.0]
    elseif bc_type == "no-penetration"
        n_len_2 = n[1]^2 + n[2]^2
        u_g = u - (u * n')*n/n_len_2 
    else
        error("bc_type  : ", bc_type )
    end
    return [state_primitive[1] ; u_g ; state_primitive[4:end] ]
end

function MoistEuler_test()
    app = MoistEuler("no-slip", "no-slip")
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