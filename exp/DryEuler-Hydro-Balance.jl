include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot


Np = 1
Nl = Np+1
Nq = ceil(Int64, (3*Np + 1)/2)
topology_type = "AtmoLES"


# Nx, Nz = 16,   32
# Lx, Lz = 16.0e3, 8.0e3

Nx, Nz = 1,   64
Lx, Lz = 2.0e3, 30.0e3


topology_size = [Lx; Lz]
topology = topology_les(Nl, Nx, Nz, Lx, Lz)
mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
gravity = true




num_state_prognostic = 4

app = DryEuler("no-penetration", nothing, "outlet", zeros(Float64, num_state_prognostic),  "periodic", nothing, "periodic", nothing, gravity)

vertical_method = "WENO3"
params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.8/Np, "dt0" => 1.0, "t_end" => 10000.00, "vertical_method" => vertical_method)
solver = Solver(app, mesh, params)


# set initial condition 
num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz

state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)

T_virt_surf, T_min_ref, H_t = 280.0, 230.0, 9.0e3
profile = init_hydrostatic_balance!(app,  mesh,  state_prognostic_0, solver.state_auxiliary_vol_l,  T_virt_surf, T_min_ref, H_t)
set_init_state!(solver, state_prognostic_0)



########3
# zz = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[1, :]
# state_primitive = solver.state_primitive
# prog_to_prim!(app, state_prognostic_0, solver.state_auxiliary_vol_l, state_primitive)
# ρ = reshape(state_primitive[:, 1 ,:], (Nl * Nx, Nz))[1, :]
# p = reshape(state_primitive[:, 4 ,:], (Nl * Nx, Nz))[1, :]
# data = reshape(state_primitive[:, 4 ,:], (Nl * Nx, Nz))[1, :]
# # PyPlot.plot(data, zz)

# Nx, Nz = 16,   32
# Lx, Lz = 16.0e3, 8.0e3

# dx, dz = Lx/Nx/Nl, Lz/Nz
# for k = 1:Nz-1
#     @info k, (p[k] -  p[k+1]),  app.g*(ρ[k+1] + ρ[k])/2.0*dz
#     @info (p[k] -  p[k+1])/dz
# end
# PyPlot.plot(p[1:4], zz[1:4], "-o")

# Δp1, Δp2, Δp3 = p[2] - p[1], p[3] - p[2], p[4] - p[3]
# Δz = zz[2] - zz[1]

# @info Δp1, Δp2, Δp3
# @show  (p[1] + p[2]) /2.0
# @show  profile(Δz)
# @show  p[1] - app.g*ρ[1]*Δz/2.0

# error("stop")

# visual(mesh, state_primitive[:, 4 ,:], "Hydro_Balance_init.png")

############

Q = solve!(solver)

state_primitive = solver.state_primitive
prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
visual(mesh, state_primitive[:,1,:], "Hydro_Balance_rho.png")
visual(mesh, state_primitive[:,2,:], "Hydro_Balance_u.png")
visual(mesh, state_primitive[:,3,:], "Hydro_Balance_w.png")
visual(mesh, state_primitive[:,4,:], "Hydro_Balance_p.png")


zz = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[1, :]
w  = reshape(state_primitive[:, 3 ,:], (Nl * Nx, Nz))[1, :]
PyPlot.plot(w, zz)
PyPlot.savefig("w.png")