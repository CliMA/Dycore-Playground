include("Mesh.jl")
include("../Apps/Application.jl")
include("DGModel.jl")
using LinearAlgebra

mutable struct Solver
    app::Application
    mesh::Mesh            
    
    
    # conservative/prognostic variables 
    # size = （Nl, num_state, nelem) 
    state_prognostic::Array{Float64,3}   # conservative variables
    Q1::Array{Float64,3}                 # conservative variables container
    
    # primitive variables 
    # size = （Nl, num_state, nelem) 
    state_primitive::Array{Float64,3}    # primitive variables container
    # diagnostic variables 
    state_diagnostic::Array{Float64,3}   # primitive variables
    
    
    
    # auxiliary variables 
    # size = （Nl, num_state, nelem) 
    state_auxiliary_vol_l::Array{Float64,3}    # auxiliary states at volume Gauss-Legendre-Lobatto points
    state_auxiliary_vol_q::Array{Float64,3}    # auxiliary states at volume Gauss-Legendre points
    state_auxiliary_surf_h::Array{Float64,4}   # auxiliary states at horizontal flux surface
    state_auxiliary_surf_v::Array{Float64,4}   # auxiliary states at vertical flux surface
    
    
    # tendency/residual variables 
    # size = （Nl, num_state, nelem) 
    tendency::Array{Float64,3}   # residual 
    k1::Array{Float64,3}         # residual  container
    k2::Array{Float64,3}         # residual  container
    k3::Array{Float64,3}         # residual  container
    k4::Array{Float64,3}         # residual  container
    
    
    time_integrator::String
    cfl_freqency::Int64
    cfl::Float64
    dt0::Float64
    t_end::Float64
    
    vertical_method::String
    
end


function Solver(app::Application, mesh::Mesh, params::Dict{String, Any})
    
    Nx, Nz, Nl, Nq = mesh.Nx,  mesh.Nz,  mesh.Nl, mesh.Nq
    num_state_prognostic, num_state_diagnostic, num_state_auxiliary = app.num_state_prognostic, app.num_state_diagnostic, app.num_state_auxiliary
    nelem = Nx * Nz
    
    state_prognostic = zeros(Float64, Nl, num_state_prognostic, nelem)
    Q1 = zeros(Float64, Nl, num_state_prognostic, nelem)
    
    # reconstruction state variables
    state_primitive  = zeros(Float64, Nl, num_state_prognostic, nelem)
    
    state_diagnostic = zeros(Float64, Nl, num_state_diagnostic, nelem)
    
    
    state_auxiliary_vol_l = zeros(Float64, Nl, num_state_auxiliary, nelem)    # auxiliary states at volume Gauss-Legendre-Lobatto points
    state_auxiliary_vol_q = zeros(Float64, Nq, num_state_auxiliary, nelem)    # auxiliary states at volume Gauss-Legendre points
    nface = 2
    state_auxiliary_surf_h = zeros(Float64, 1,  num_state_auxiliary, nface, nelem)    # auxiliary states at horizontal flux surface
    state_auxiliary_surf_v = zeros(Float64, Nl, num_state_auxiliary, nface, nelem)   # auxiliary states at vertical flux surface
    
    
    
    init_state_auxiliary!(app, mesh, 
    state_auxiliary_vol_l, state_auxiliary_vol_q, 
    state_auxiliary_surf_h, state_auxiliary_surf_v)
    
    tendency = zeros(Float64, Nl, num_state_prognostic, nelem)
    k1 = zeros(Float64, Nl, num_state_prognostic, nelem)
    k2 = zeros(Float64, Nl, num_state_prognostic, nelem)
    k3 = zeros(Float64, Nl, num_state_prognostic, nelem)
    k4 = zeros(Float64, Nl, num_state_prognostic, nelem)
    
    time_integrator =  params["time_integrator"]
    cfl_freqency = params["cfl_freqency"]
    cfl = params["cfl"]
    dt0 = params["dt0"]
    t_end = params["t_end"]
    
    vertical_method = params["vertical_method"]
    
    
    Solver(app, mesh, 
    state_prognostic, Q1, 
    state_primitive, 
    state_diagnostic,
    state_auxiliary_vol_l, state_auxiliary_vol_q, state_auxiliary_surf_h, state_auxiliary_surf_v,
    tendency, k1, k2, k3, k4, 
    time_integrator, cfl_freqency, cfl, dt0, t_end, 
    vertical_method
    )
    
end

function set_init_state!(solver::Solver, state_prognostic_0::Array{Float64, 3})
    solver.state_prognostic .= state_prognostic_0 
end


# W0, initial condition (conservative state variables)
# methods: dictionary for method parameters
# T: end time
function solve!(solver::Solver)
    
    app, mesh = solver.app,  solver.mesh
    
    Q = solver.state_prognostic
    
    t, t_end = 0.0, solver.t_end
    
    dt0, cfl, cfl_freqency = solver.dt0, solver.cfl, solver.cfl_freqency
    Q_aux = solver.state_auxiliary_vol_l
    
    cfl_dt0 = compute_cfl_dt(solver.app, solver.mesh, Q, Q_aux, cfl)
    
    dt = min(dt0, cfl_dt0)
    update_sponge_params!(app, dt)
    
    @info "dt , dt0, cfl_dt0 = ", dt , dt0, cfl_dt0
    
    ite = 0
    
    while t < t_end
        # compute time step size
        ite += 1
        if cfl_freqency > 0 && ite%cfl_freqency == 0
            dt = compute_cfl_dt(solver.app, solver.mesh, Q, Q_aux, cfl)
            update_sponge_params!(app, dt)
        end
        
        if dt + t > t_end
            dt = t_end - t
            update_sponge_params!(app, dt)
        end
        
        # update solution in W for the next time step 
        time_advance!(solver, Q, dt)
        
        # apply filter
        apply_filter(Q)
        
        t += dt
        
    end
    
    @info "Finish simulation: total iterations: ", ite, " dt0, cfl_dt0, t_end = ", dt0, " ", cfl_dt0, " ", t_end
    
    return Q
end






# advance to next time step
# W: current conservative state
# update W to the state at the next time state
function time_advance!(solver::Solver,  Q::Array{Float64,3}, dt::Float64)
    
    app, mesh = solver.app, solver.mesh
    time_integrator = solver.time_integrator
    
    Q1 = solver.Q1
    
    dQ = solver.tendency
    k1, k2, k3, k4 = solver.k1, solver.k2, solver.k3, solver.k4
    
    if time_integrator == "RK2"
        
        k1 .= 0
        spatial_residual!(solver, Q, k1)
        
        Q1 .= Q + k1 .* dt
        
        k2 .= 0
        spatial_residual!(solver, Q1, k2);
        
        dQ .= 1.0 / 2.0 * (k1 + k2)
        Q .+=  dQ .* dt
        
    elseif time_integrator == "RK4"
        
        k1 .= 0
        spatial_residual!(solver, Q, k1)
        
        Q1 .= Q + k1 .* dt/2.0
        k2 .= 0
        spatial_residual!(solver, Q1, k2);
        
        Q1 .= Q + k2 .* dt/2.0
        k3 .= 0
        spatial_residual!(solver, Q1, k3);
        
        Q1 .= Q + k3 .* dt
        k4 .= 0
        spatial_residual!(solver, Q1, k4);
        
        
        dQ .= 1.0 / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        Q .+=  dQ .* dt
        
        
    else
        error("Time integrator ", method, "has not implemented")
    end
    
end


# Compute spatial residual vector on each node 
# dQ/dt = R
function spatial_residual!(solver::Solver, Q::Array{Float64,3}, dQ::Array{Float64,3})
    
    dQ .= 0.0
    app, mesh = solver.app, solver.mesh
    state_auxiliary_vol_l  =  solver.state_auxiliary_vol_l     
    state_auxiliary_vol_q  =  solver.state_auxiliary_vol_q   
    state_auxiliary_surf_h =  solver.state_auxiliary_surf_h   
    state_auxiliary_surf_v =  solver.state_auxiliary_surf_v
    
    state_primitive = solver.state_primitive
    prog_to_prim!(app, Q, state_auxiliary_vol_l,  state_primitive)
    update_state_auxiliary!(app, mesh, state_primitive , state_auxiliary_vol_l, state_auxiliary_vol_q, state_auxiliary_surf_h, state_auxiliary_surf_v)
    
    compute_min_max(app, state_primitive)
    
    
    
    horizontal_volume_tendency!(app, mesh, Q, state_auxiliary_vol_l, state_auxiliary_vol_q, dQ)
    #horizontal_volume_tendency!(app, mesh, Q, state_auxiliary_vol_q, dQ)
    #@show "horizontal_volume_tendency! ", norm(dQ)
    
    #@show "horizontal_volume_tendency! ", norm(dQ[:,1,:]), norm(dQ[:,2,:]), norm(dQ[:,3,:]), norm(dQ[:,4,:])
    
    horizontal_interface_tendency!(app, mesh, Q, state_auxiliary_surf_h, dQ)
    #@show "horizontal_interface_tendency! ", norm(dQ)
    
    #@show "horizontal_interface_tendency! ", norm(dQ[:,1,:]), norm(dQ[:,2,:]), norm(dQ[:,3,:]), norm(dQ[:,4,:])


    vertical_interface_tendency!(app, mesh, state_primitive, state_auxiliary_vol_l, state_auxiliary_surf_v, dQ; method = solver.vertical_method)
    #@show "vertical_interface_tendency! ", norm(dQ)
    
    #@show "vertical_interface_tendency! ", norm(dQ[:,1,:]), norm(dQ[:,2,:]), norm(dQ[:,3,:]), norm(dQ[:,4,:])
    
    source_tendency!(app, mesh, Q, state_auxiliary_vol_l, dQ)
    
    # @info dQ[:, 3, :]
    @show "source_tendency! ", norm(dQ[:,1,:]), norm(dQ[:,2,:]), norm(dQ[:,3,:]), norm(dQ[:,4,:])
    
    
    # error("stop")
    
    M_lumped = @view mesh.vol_l_geo[3, :, :]
    for s = 1:app.num_state_prognostic
        dQ[:,s,:] ./= M_lumped
    end


    # @info dQ[:, 3, :]
    # @info dQ[1, 3, :]

    # @show "final! ", norm(dQ[:,1,:]), norm(dQ[:,2,:]), norm(dQ[:,3,:]), norm(dQ[:,4,:])

    # error("stop")
    
end


function apply_filter(Q::Array{Float64,3})

    return ;

    Nl, num_state_prognostic, nelem = size(Q)
    
    Np = Nl - 1
    ξl, ωl = lglpoints(Np)
    filter = CutoffFilter(ξl, Np)
    
    
    filter_states = [2,3]
    for e = 1:nelem
        for s in filter_states
            Q[:, s, e] .= filter.filter * Q[:, s, e] 
        end
    end
    
    
end

function compute_cfl_dt(app::Application, mesh::Mesh, Q::Array{Float64,3}, Q_aux::Array{Float64,3}, cfl::Float64)
    
    @assert(cfl >= 0.0)
    
    Δs_min = mesh.Δs_min
    dim, Nl, nelem = size(Δs_min)
    
    # compute time step based on cfl number 
    dt_h, dt_v = Inf64, Inf64
    
    for e = 1:nelem
        
        for il = 1:Nl
            u = compute_wave_speed(app, Q[il, :, e], Q_aux[il, :, e])
            
            
            
            dt_h = min(dt_h, cfl * Δs_min[1, il, e]/u)
            
            dt_v = min(dt_v, cfl * Δs_min[2, il, e]/u)
        end
    end
    @info "compute_cfl_dt, dt = ", min(dt_h, dt_v)
    return min(dt_h, dt_v)
end


