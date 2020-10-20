"""
2D Constant velocity advection equation

da/dt + ∇(a u) = 0
"""
mutable struct Adv <: Application

    
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

    hydrostatic_balance::Bool

    # constant advection velocity
    u::Float64
    w::Float64

    
    # Lax Friedrichs flux dissipation parameter
    α::Float64


    Δt::Float64
    zT::Float64
    zD::Float64
    xT::Float64
    xD::Float64
    a_sponge::Float64

end

function Adv(bc_bottom_type::String,  bc_bottom_data::Union{Array{Float64, 1}, Nothing},
             bc_top_type::String,     bc_top_data::Union{Array{Float64, 1}, Nothing},
             bc_left_type::String,    bc_left_data::Union{Array{Float64, 1}, Nothing},
             bc_right_type::String,   bc_right_data::Union{Array{Float64, 1}, Nothing},
             u::Float64, w::Float64,  α::Float64 = 0.0, hydrostatic_balance::Bool=false)
    num_state_prognostic = 1
    num_state_diagnostic = 1
    num_state_auxiliary = 0

    Δt, zT, zD, xT, xD, a_sponge = -1.0, -1.0, Inf64, -1.0, Inf64, 0.0

    Adv(num_state_prognostic, num_state_diagnostic, num_state_auxiliary,
    bc_bottom_type, bc_bottom_data,
    bc_top_type, bc_top_data,
    bc_left_type, bc_left_data,
    bc_right_type, bc_right_data,
    hydrostatic_balance,
    u, w, α,
    Δt, zT, zD, xT, xD, a_sponge)
end


function update_sponge_params!(app::Adv, Δt::Float64=app.Δt, zT::Float64=app.zT, zD::Float64=app.zD, 
    xT::Float64=app.xT, xD::Float64=app.xD, a_sponge::Float64 =app.a_sponge)
    app.Δt, app.zT, app.zD, app.a_sponge = Δt, zT, zD, a_sponge
    app.xT, app.xD = xT, xD
end


function compute_wave_speed(app::Adv, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    return sqrt(app.u^2 + app.w^2)
end

function prog_to_prim(app::Adv, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    return state_prognostic
end

function prim_to_prog(app::Adv, state_primitive::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    return state_primitive
end

# function prim_to_prog!(app::Adv, state_primitive::Array{Float64, 2}, state_prognostic::Array{Float64, 2})
#     state_prognostic .= state_primitive 
# end

function prog_to_prim!(app::Adv, state_prognostic::Array{Float64, 3}, state_auxiliary::Array{Float64, 3}, state_primitive::Array{Float64, 3})
    state_primitive .= state_prognostic
end



function flux_first_order(app::Adv, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1})
    
    return [state_prognostic*app.u  state_prognostic*app.w]
end

# Lax Friedrichs flux
function numerical_flux_first_order(app::Adv, state_prognostic⁻::Array{Float64, 1}, state_auxiliary⁻::Array{Float64, 1}, 
    state_prognostic⁺::Array{Float64, 1}, state_auxiliary⁺::Array{Float64, 1}, 
    n::Array{Float64, 1})
    un = app.u*n[1] + app.w*n[2]
    
    α = app.α
    
    return  0.5*(state_prognostic⁻ + state_prognostic⁺) * un - abs(un) * (1 - α)/2.0 * (state_prognostic⁺ - state_prognostic⁻)
    
end

function source(app::Adv, state_prognostic::Array{Float64, 1}, state_auxiliary::Array{Float64, 1}, x::Float64)
    #no source, no sponge
    return zeros(Float64, app.num_state_prognostic)
end


# initialize 
function init_state!(app::Adv, mesh::Mesh, state_prognostic::Array{Float64, 3}, func::Function)

    Nl, num_state_prognostic, nelem = size(state_prognostic)
    vol_l_geo = mesh.vol_l_geo
    
    for e = 1:nelem
        for il = 1:Nl

            x, z = vol_l_geo[1:2, il, e]
            
            state_prognostic[il, 1, e] = func(x, z)
        end
    end
end

function init_state_auxiliary!(app::Adv, mesh::Mesh, 
    state_auxiliary_vol_l::Array{Float64, 3}, state_auxiliary_vol_q::Array{Float64, 3}, 
    state_auxiliary_surf_h::Array{Float64, 4}, state_auxiliary_surf_v::Array{Float64, 4})
    
end


function bc_impose(app::Adv, state_primitive::Array{Float64, 1}, bc_type::String, n::Array{Float64, 1})
    @warn("Adv bc_impose")
    return state_primitive
end



function compute_min_max(app::Adv, state_primitive::Array{Float64,3})
    
    @info "min a = ", minimum(state_primitive[:,1,:]), " max a = ", maximum(state_primitive[:,1,:])

end