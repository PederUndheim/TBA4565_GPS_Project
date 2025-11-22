import numpy as np
import math

# WGS84 ellipsoid constants
WGS84_A = 6378137.0                   # a (semi-major axis) [m]
WGS84_F = 1 / 298.257223563           # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)   # b (semi-minor axis) [m]

def geodetic_to_cartesian(lat_deg, lon_deg, h_m):
    
    # Convert degrees to radians
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    a = WGS84_A
    b = WGS84_B

    N = a**2 / math.sqrt(a**2 * math.cos(lat_rad)**2 + b**2 * math.sin(lat_rad)**2)

    # Cartesian coordinates
    X = (N + h_m) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + h_m) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = (b**2 / a**2 * N + h_m) * math.sin(lat_rad)

    return np.array([X, Y, Z])



def cartesian_to_geodetic(X, Y, Z):
    a = WGS84_A
    b = WGS84_B
    e2 = (a**2 - b**2) / a**2 

    p = math.sqrt(X**2 + Y**2)

    lat = math.atan2(Z, p * (1 - e2))

    for _ in range(20):
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        N = a**2 / math.sqrt(a**2 * cos_lat**2 + b**2 * sin_lat**2)
        h = p / cos_lat - N
        lat_new = math.atan2(Z, p * (1 - e2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    lon = math.atan2(Y, X)
    N = a**2 / math.sqrt(a**2 * math.cos(lat)**2 + b**2 * math.sin(lat)**2)
    h = p / math.cos(lat) - N

    return (math.degrees(lat), math.degrees(lon), h)


