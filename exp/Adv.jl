include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

"""
periodicity T = 1
"""
function adv_square(test_case::String, vertical_method::String, t_end::Float64=3.0, Np::Int64=3, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    
    Nl = Np+1
    
    Nz = 50
    Nx = ceil(Int64, Nz/Nl)
    Lx, Lz = 3.0, 3.0
    
    topology_type = "AtmoLES"
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    
    
    
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    app = Adv("periodic", nothing,  # bottom
    "periodic", nothing,  # top
    "periodic", nothing,  # left
    "periodic", nothing,  # right
    1.0, 1.0)
    
    dt = 0.1
    params = Dict("time_integrator" => "RK4", "cfl_freqency" => -1,  "cfl" => 1/Np, "dt0" => dt, "t_end" => t_end, "vertical_method" => vertical_method)
    solver = Solver(app, mesh, params)
    
    # initial condition
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
    
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    
    const_func = (x::Float64, z::Float64) -> 1.0
    function square_2d_func(x::Float64, z::Float64)
        if (x > -Lx/6.0 && x < Lx/6.0) && (z > Lz/3.0 && z < 2.0*Lz/3.0)
            return 1.0
        else
            return 0.0
        end
    end
    
    function gaussian_2d_func(x::Float64, z::Float64)
        
        μx1, μz1, σ1 = 0.5*Lx - Lx/2.0, 0.5*Lz, 0.1*Lx
        
        return exp.(-(((x - μx1)/σ1)^2 + ((z - μz1)/σ1)^2)/2.0)/(σ1*sqrt(2.0*pi)) 
    end
    
    if test_case == "square"
        init_state!(app, mesh, state_prognostic_0, square_2d_func)
    elseif test_case == "gaussian"
        init_state!(app, mesh, state_prognostic_0, gaussian_2d_func)
    else
        error("initial condition : ", test_case, " has not implemented yet")
    end
    
    set_init_state!(solver, state_prognostic_0)
    
    # visualize the initial condition 
    visual(mesh, state_prognostic_0[:,1,:], "Adv_"*test_case*"_init.png")
    
    Q = solve!(solver)
    
    # visualize the solution at t_end 
    visual(mesh, Q[:,1,:], "Adv_"*test_case*"_end_"*vertical_method*".png")
    
    coord_z = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[div(Nl * Nx, 2), :]
    Q0_z  = reshape(state_prognostic_0, (Nl * Nx, Nz))[div(Nl * Nx, 2), :]
    Q_z  = reshape(Q, (Nl * Nx, Nz))[div(Nl * Nx, 2), :]
    coord_x = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[:, div(Nz, 2)] .+ Lx/2.0
    Q_x  = reshape(Q, (Nl * Nx, Nz))[:, div(Nz, 2)]
    
    PyPlot.figure()
    PyPlot.plot(coord_z, Q0_z, "-o", fillstyle = "none", label = "Ref")
    PyPlot.plot(coord_z, Q_z, "-o", fillstyle = "none", label = vertical_method)
    PyPlot.plot(coord_x, Q_x, "-o", fillstyle = "none", label = "DG (p=3 overintegration)")
    PyPlot.legend()
    PyPlot.savefig("Adv_"*test_case*"_end_section"*vertical_method*".pdf")
    
    
    return 
    
    
end

period = 1
adv_square("square", "WENO3", 3.0*period)
adv_square("square", "WENO5", 3.0*period)
adv_square("square", "FV",    3.0*period)

adv_square("gaussian", "WENO3", 3.0*period)
adv_square("gaussian", "WENO5", 3.0*period)
adv_square("gaussian", "FV",    3.0*period)





