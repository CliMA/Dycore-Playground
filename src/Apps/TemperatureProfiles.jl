struct DecayingTemperatureProfile
    "Virtual temperature at surface (K)"
    T_virt_surf::Float64
    "Minimum virtual temperature at the top of the atmosphere (K)"
    T_min_ref::Float64
    "Height scale over which virtual temperature drops (m)"
    H_t::Float64
    
    Rd::Float64  
    g::Float64 
    MSLP::Float64  
    
end

function DecayingTemperatureProfile(app::Application, T_virt_surf::Float64, T_min_ref::Float64, H_t::Float64)
    
    Rd, g, MSLP = app.Rd, app.g, app.MSLP
    return DecayingTemperatureProfile(T_virt_surf, T_min_ref, H_t, Rd, g, MSLP)
end

function (profile::DecayingTemperatureProfile)(
    z::Float64,
    ) 
    _R_d  = profile.Rd
    _grav = profile.g
    _MSLP = profile.MSLP

    # todo
    
    # Scale height for surface temperature
    H_sfc = _R_d * profile.T_virt_surf / _grav
    H_t = profile.H_t
    z′ = z / H_t
    tanh_z′ = tanh(z′)
    
    ΔTv = profile.T_virt_surf - profile.T_min_ref
    Tv = profile.T_virt_surf - ΔTv * tanh_z′
    
    ΔTv′ = ΔTv / profile.T_virt_surf
    p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
    p /= H_sfc * (1 - ΔTv′^2)
    p = _MSLP * exp(p)
    ρ = p/(_R_d*Tv)


    # ρ = 1.0
    # p = _MSLP - _grav*ρ*z
    # Tv = NaN64
    return (Tv, p, ρ)
end