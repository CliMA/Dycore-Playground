include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")



Np = 3
Nl = Np+1
Nq = ceil(Int64, (3*Np + 1)/2)

Nx, Nz = 8, 32
Lx, Lz = 3.0, 3.0

topology_type = "AtmoLES"
topology_size = [Lx; Lz]
topology = topology_les(Nl, Nx, Nz, Lx, Lz)



mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
app = Adv("periodic", "periodic", 1.0, 1.0)

dt = 0.1
params = Dict("Time_Integrator" => "RK2", "cfl_freqency" => -1,  "cfl" => 1/Np, "dt0" => dt, "t_end" => 3.0)
solver = Solver(app, mesh, params)

# initial condition
num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz


state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)

const_func = (x::Float64, z::Float64) -> 1.0
function square_2d_func(x::Float64, z::Float64)
    @info x, z
    if (x > -Lx/6.0 && x < Lx/6.0) && (z > Lz/3.0 && z < 2.0*Lz/3.0)
        return 1.0
    else
        return 0.0
    end
end
init_state!(app, mesh, state_prognostic_0, square_2d_func)

set_init_state!(solver, state_prognostic_0)

visual(mesh, state_prognostic_0[:,1,:], "Adv_init.png")


Q = solve!(solver)
visual(mesh, Q[:,1,:], "Adv_end.png")



