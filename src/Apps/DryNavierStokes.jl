include("DryEuler.jl")

mutable struct DryNavierStokes <: Application
    num_state_prognostic::Int64
    num_state_diagnostic::Int64
    num_state_auxiliary::Int64

    # new 
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



function DryNavierStokes(bc_bottom_type::String,  bc_bottom_data::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String,     bc_top_data::Union{Array{Float64, 1}, Nothing},
    bc_left_type::String,    bc_left_data::Union{Array{Float64, 1}, Nothing},
    bc_right_type::String,   bc_right_data::Union{Array{Float64, 1}, Nothing},
    gravity::Bool, hydrostatic_balance::Bool)
    
    num_state_prognostic = 4
    num_state_diagnostic = 4
    # Φ ∇Φ ρ_ref p_ref
    num_state_auxiliary = 5

    num_state_gradient = 3
    
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
    
    
    DryNavierStokes(num_state_prognostic, num_state_diagnostic, num_state_auxiliary, num_state_gradient,
    bc_bottom_type, bc_bottom_data,
    bc_top_type, bc_top_data,
    bc_left_type, bc_left_data,
    bc_right_type, bc_right_data,
    hydrostatic_balance,
    g, γ, Rd, MSLP,
    Δt, zT, zD, 
    xT, xD, u_sponge)

end