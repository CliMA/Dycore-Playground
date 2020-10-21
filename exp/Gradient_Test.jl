include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")


"""
return f, ∂f/∂x, ∂f/∂z
"""
function sin_cos_func(x::Float64, z::Float64)
    return sin(2*x*pi)*cos(4*z*pi), 2*pi*cos(2*x*pi)*cos(4*z*pi), -4*pi*sin(2*x*pi)*sin(4*z*pi)
end

function const_func(x::Float64, z::Float64)
    return 1.0, 0.0, 0.0
end

function linear_func(x::Float64, z::Float64)
    return x + 2*y, 1.0, 2.0
end

function gradient_test(test_type::String, Np::Int64=2, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    dim  = 2 
    Nl = Np+1
    
    if test_type == "AtmoLES"
        

        init_func = sin_cos_func
        
        topology_type = "AtmoLES"
        Nx, Nz = 32,   32*Nl
        Lx, Lz = 1.0,  1.0
        
        
        topology_size = [Lx, Lz]
        topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    elseif test_type == "AtmoGCM"
        

        init_func = sin_cos_func


        topology_type = "AtmoGCM"
        Nx, Nz = 64,   64*Nl
        r,  R  = 1.0,  2.0
        
        
        topology_size = [r, R]
        topology = topology_gcm(Nl, Nx, Nz, r, R)


    elseif test_type == "AtmoLES-Mountain"
    else
        return
    end
    
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    vol_l_geo = mesh.vol_l_geo
    gravity = false
    hydrostatic_balance = false
    
    # set initial condition 
    num_state_gradient, nelem = 3, Nx*Nz
    
    
    
    state_gradient = zeros(Nl, num_state_gradient, nelem)
    ∇state_gradient_ref = zeros(Nl, num_state_gradient, nelem, 2)
    for ix = 1:Nx
        for iz = 1:Nz
            for il = 1:Nl
                e = ix + (iz - 1)*Nx
                x, z = vol_l_geo[1:2, il, e]
                u, ∂u∂x, ∂u∂z = init_func(x, z)
                
                state_gradient[il, :, e] .= u
                ∇state_gradient_ref[il, :, e, 1] .= ∂u∂x
                ∇state_gradient_ref[il, :, e, 2] .= ∂u∂z
            end
        end
    end
    
    ∇ref_state_gradient = zeros(Nl, num_state_gradient, nelem, dim)
    ∇state_gradient = zeros(Nl, num_state_gradient, nelem, dim)
    
    app = DryNavierStokes("periodic", nothing, "periodic", nothing,  "periodic", nothing, "periodic", nothing, gravity, hydrostatic_balance)
    
    compute_gradients!(app, mesh, state_gradient, ∇ref_state_gradient, ∇state_gradient)
    
    vmin, vmax = -2π, 2π
    visual(mesh, ∇state_gradient_ref[:, 1, :, 1], test_type*"DuDx_ref.png", vmin, vmax)
    visual(mesh, ∇state_gradient[:, 1, :, 1],     test_type*"DuDx.png", vmin, vmax)
    
    vmin, vmax = -4π, 4π
    visual(mesh, ∇state_gradient_ref[:, 1, :, 2], test_type*"DuDy_ref.png", vmin, vmax)
    visual(mesh, ∇state_gradient[:, 1, :, 2],     test_type*"DuDy.png", vmin, vmax)
    
end


test_type = "AtmoLES"
Np = 2
gradient_test(test_type, Np)


test_type = "AtmoGCM"
Np = 3
gradient_test(test_type, Np)