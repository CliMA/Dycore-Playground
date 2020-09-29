include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")




function shock_tube(direction::String)
    @assert(direction == "horizontal" || direction == "vertical")
    
    Np = 4
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)
    topology_type = "AtmoLES"
    
    if direction == "vertical"
        
        Nx, Nz = 2,   100
        Lx, Lz = 1.0, 1.0

    else direction == "horizontal"

        Nx, Nz = 20,  1
        Lx, Lz = 1.0, 1.0
        
    end
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    gravity = false
    
    if direction == "vertical"
        app = DryEuler("no-slip", nothing, "no-slip", nothing,  "periodic", nothing, "periodic", nothing, gravity)
    else direction == "horizontal"
        app = DryEuler("periodic", nothing, "periodic", nothing, "no-slip", nothing, "no-slip", nothing,  gravity)
    end
    
    
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 1/Np, "dt0" => 0.1, "t_end" => 0.2)
    solver = Solver(app, mesh, params)
    
    
    # set initial condition 
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    prim_l = [1.0;   0.0; 0.0; 1.0]
    prim_r = [0.125; 0.0; 0.0; 0.1]
    cons_l = prim_to_prog(app, prim_l, zeros(Float64, app.num_state_auxiliary))
    cons_r = prim_to_prog(app, prim_r, zeros(Float64, app.num_state_auxiliary))
    
    function shock_tube_func(x::Float64, z::Float64)
        if direction == "vertical"
            if z  <= Lz/2.0
                return cons_l
            else
                return cons_r
            end
        else
            if x  <= 0.0
                return cons_l
            else
                return cons_r
            end
        end
    end
    init_state!(app, mesh, state_prognostic_0, shock_tube_func)
    set_init_state!(solver, state_prognostic_0)
    
    # visual(mesh, state_prognostic_0[:,1,:], "Sod_init_"*direction*".png")
    
    
    Q = solve!(solver)
    
    
    visual(mesh, Q[:,1,:], "Sod_init_"*direction*".png")
    
    
end


shock_tube("vertical")

shock_tube("horizontal")