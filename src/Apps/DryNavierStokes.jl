include("DryEuler.jl")

mutable struct DryNavierStokes <: DryEuler
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