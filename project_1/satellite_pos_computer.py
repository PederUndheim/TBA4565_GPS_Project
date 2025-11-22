import numpy as np

GM = 3.986005e14            # m^3/s^2
OMEGA_E = 7.2921151467e-5   # rad/s

# Solve Kepler's equation with Newton-Raphson method
def kepler_newton_raphson(Mk, e):
    E = Mk
    for _ in range(3):
        E = E + (Mk - E + e*np.sin(E)) / (1 - e*np.cos(E))
    return E

# Compute satellite position in ECEF coordinates
def sat_position(params, t_s, no_corrections=False):
    toe   = params["TransTime"]
    sqrtA = params["sqrtA"]
    a     = sqrtA**2
    e     = params["Eccentricity"]
    M0    = params["M0"]
    Delta_n  = params["DeltaN"]
    omega    = params["omega"]
    i0       = params["Io"]
    Omega0   = params["Omega0"]
    IDOT     = params["IDOT"]
    OmegaDot = params["OmegaDot"]
    Cuc, Cus = params["Cuc"], params["Cus"]
    Crc, Crs = params["Crc"], params["Crs"]
    Cic, Cis = params["Cic"], params["Cis"]

    if no_corrections:
        Cuc = Cus = Crc = Crs = Cic = Cis = 0.0
        Delta_n = IDOT = OmegaDot = 0.0


    # Time from ephemeris reference
    tk = t_s - toe
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800

    n0 = np.sqrt(GM / a**3)
    n = n0 + Delta_n

    Mk = M0 + n * tk
    Ek = kepler_newton_raphson(Mk, e)

    fk = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(Ek / 2))

    phi = omega + fk
    u = phi + Cuc * np.cos(2*phi) + Cus * np.sin(2*phi)
    r = a * (1 - e*np.cos(Ek)) + Crc * np.cos(2*phi) + Crs * np.sin(2*phi)
    i = i0 + IDOT*tk + Cic * np.cos(2*phi) + Cis * np.sin(2*phi)

    Omega = Omega0 + (OmegaDot - OMEGA_E)*tk - OMEGA_E*toe

    # Position in orbital plane
    x_orb = r * np.cos(u)
    y_orb = r * np.sin(u)

    # ECEF coordinates
    X = x_orb * np.cos(Omega) - y_orb * np.cos(i) * np.sin(Omega)
    Y = x_orb * np.sin(Omega) + y_orb * np.cos(i) * np.cos(Omega)
    Z = y_orb * np.sin(i)

    return np.array([X, Y, Z])