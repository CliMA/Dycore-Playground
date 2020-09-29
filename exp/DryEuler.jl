include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")



Np = 3
Nl = Np+1
Nq = ceil(Int64, (3*Np + 1)/2)

Nx, Nz = 8,100
Lx, Lz = 1.0, 1.0

topology_type = "AtmoLES"
topology_size = [Lx; Lz]
topology = topology_les(Nl, Nx, Nz, Lx, Lz)



mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
gravity = false
app = DryEuler("periodic", "periodic", gravity)



params = Dict("Time_Integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 1/Np, "dt0" => 0.1, "t_end" => 0.2)
solver = Solver(app, mesh, params)




# set initial condition 
num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz

state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
prim_l = [1.0;   0.0; 0.0; 1.0]
prim_r = [0.125; 0.0; 0.0; 0.1]
cons_l = prim_to_prog(app, prim_l, zeros(Float64, app.num_state_auxiliary))
cons_r = prim_to_prog(app, prim_r, zeros(Float64, app.num_state_auxiliary))

function shock_tube_func(x::Float64, z::Float64)
    if z  <= Lz/2.0
        return cons_l
    else
        return cons_r
    end
end
init_state!(app, mesh, state_prognostic_0, shock_tube_func)
set_init_state!(solver, state_prognostic_0)

visual(mesh, state_prognostic_0[:,1,:], "Sod_init.png")


Q = solve!(solver)


visual(mesh, Q[:,1,:], "Sod_end.png")


