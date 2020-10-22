include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")




function isentropic_vortex(vertical_method::String, Np::Int64=2, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    
    Nl = Np+1
    topology_type = "AtmoLES"
    
    
    Nx, Nz = 32,   32*Nl
    Lx, Lz = 10.0, 10.0
    
    
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    # viscous, ν, Pr = false, NaN64, NaN64
    viscous, ν, Pr = true, 0.01, 0.72
    gravity = false
    hydrostatic_balance = false
    
    # set initial condition 
    num_state_prognostic, nelem = 4, Nx*Nz
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    
    
    app = DryAtmo("periodic", nothing, "periodic", nothing,  "periodic", nothing, "periodic", nothing, 
    viscous, ν, Pr,  
    gravity, hydrostatic_balance)
    
    
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.5/Np, "dt0" => 0.02, "t_end" => 5.0000, "vertical_method" => vertical_method)
    
    solver = Solver(app, mesh, params)
    
    
    function init_func(x::Float64, z::Float64, t::Float64 = 0.0)
        
        γ = 1.4  
        ρ∞ = 1.0
        
        u∞ = 1.0
        M∞ = 0.5
        c∞ = u∞/M∞ 
        θ = atan(0.5)
        p∞ = ρ∞*c∞^2/γ
        ū, v̄ = u∞*cos(θ), u∞*sin(θ)
        ϵ = 0.3
        r = 1.5
        x₀, z₀ = 0.0, 5.0
        
        
        f(x, z, t) = (1 - (x - x₀ - ū*t)^2 - (z - z₀ - v̄*t)^2)/r^2
        u = u∞*(cos(θ) - ϵ*((z -z₀) - v̄*t)/(2*π*r)*exp(f(x,z,t)/2.0))
        v = u∞*(sin(θ) + ϵ*((x -x₀) - ū*t)/(2*π*r)*exp(f(x,z,t)/2.0))
        ρ = ρ∞*(1 - ϵ^2*(γ-1)*M∞^2/(8.0*π^2)*exp(f(x,z, t)))^(1/(γ-1))
        p = p∞*(1 - ϵ^2*(γ-1)*M∞^2/(8.0*π^2)*exp(f(x,z, t)))^(γ/(γ-1))

        ρe = p/(γ-1) + 0.5*(ρ*u^2 + ρ*v^2) 
        return [ρ; ρ*u; ρ*v; ρe]
    end
    
    init_state!(app, mesh, state_prognostic_0, init_func)
    set_init_state!(solver, state_prognostic_0)
    

    state_primitive_0 = similar(state_prognostic_0)
    prog_to_prim!(app, state_prognostic_0, solver.state_auxiliary_vol_l, state_primitive_0)
    visual(mesh, state_primitive_0[:,1,:], "Isentropic_Vortex_rho_init.png")
    visual(mesh, state_primitive_0[:,2,:], "Isentropic_Vortex_u_init.png")
    visual(mesh, state_primitive_0[:,3,:], "Isentropic_Vortex_v_init.png")
    visual(mesh, state_primitive_0[:,4,:], "Isentropic_Vortex_p_init.png")
    
    
    Q = solve!(solver)
    
    
    state_primitive = similar(Q)
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    visual(mesh, state_primitive[:,1,:], "Isentropic_Vortex_rho_end.png")
    visual(mesh, state_primitive[:,2,:], "Isentropic_Vortex_u_end.png")
    visual(mesh, state_primitive[:,3,:], "Isentropic_Vortex_v_end.png")
    visual(mesh, state_primitive[:,4,:], "Isentropic_Vortex_p_end.png")
    
    
end


vertical_method = "FV"
Np = 3
isentropic_vortex(vertical_method, Np)

