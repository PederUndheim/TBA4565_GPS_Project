import pandas as pd
import numpy as np
from nav_file_reader import read_nav_file
from satellite_pos_computer import sat_position
from coord_transformer import geodetic_to_cartesian, cartesian_to_geodetic
from LS_computer import least_squares_position

gps_data_path = "project_1/data/gps_data.txt"

c = 299792458.0  # m/s
satellites = ["G08", "G10", "G21", "G24", "G17", "G03", "G14"]

# Pseudorange P(L1) [m] from the project table
P_L1 = {
    "G08": 22550792.660,
    "G10": 22612136.900,
    "G21": 20754631.240,
    "G24": 23974471.500,
    "G17": 24380357.760,
    "G03": 24444143.500,
    "G14": 22891323.280
}

# Satellite clock error dt_j [s] from the project table
DT = {
    "G08": +0.00013345632,
    "G10": +0.000046155711,
    "G21": -0.00015182034,
    "G24": +0.00026587520,
    "G17": -0.00072144074,
    "G03": +0.00022187057,
    "G14": -0.00013020719
}

# Ionosphere and troposphere corrections [m]
DION = {
    "G08": 3.344, "G10": 2.947, "G21": 2.505, "G24": 3.644,
    "G17": 6.786, "G03": 4.807, "G14": 4.598,
}
DTROP = {
    "G08": 4.055, "G10": 4.297, "G21": 2.421, "G24": 9.055,
    "G17": 9.756, "G03": 10.863, "G14": 4.997,
}

# Initial approximate receiver geodetic coordinates (task text)
receiver_lat0 = 63.2     # degrees
receiver_lon0 = 10.2     # degrees
receiver_h0   = 100.0    # meters





if __name__ == "__main__":

    # Task 1: Compute satellite coordinates
    nav = read_nav_file(gps_data_path)
    T = 558000.0
    sat_positions = {}

    for sv in satellites:
        params = nav[sv][0][1]
        t_s = T - P_L1[sv] / c + DT[sv]
        sat_positions[sv] = sat_position(params, t_s, no_corrections=False)

    print("----------------------------------------------------------------------------------------------")
    print("\n=== Satellite Positions (ECEF) with Corrections ===\n")
    for sv in satellites:
        X, Y, Z = sat_positions[sv]
        print(f"{sv}:")
        print(f"   X = {X:14.3f} m")
        print(f"   Y = {Y:14.3f} m")
        print(f"   Z = {Z:14.3f} m\n")
    print("----------------------------------------------------------------------------------------------")





    # Task 2: Compute satellite coordinates without corrections
    sat_positions_no_corr = {}
    for sv in satellites:
        params = nav[sv][0][1]
        t_s = T - P_L1[sv] / c + DT[sv]
        sat_positions_no_corr[sv] = sat_position(params, t_s, no_corrections=True)
    
    print("\n=== Satellite Positions (ECEF) without Corrections ===\n")
    for sv in satellites:
        X, Y, Z = sat_positions_no_corr[sv]
        print(f"{sv}:")
        print(f"   X = {X:14.3f} m")
        print(f"   Y = {Y:14.3f} m")
        print(f"   Z = {Z:14.3f} m\n")
    
    print("\n=== Difference: WITH − WITHOUT Corrections ===\n")
    for sv in satellites:
        dx = sat_positions[sv][0] - sat_positions_no_corr[sv][0]
        dy = sat_positions[sv][1] - sat_positions_no_corr[sv][1]
        dz = sat_positions[sv][2] - sat_positions_no_corr[sv][2]
        print(f"{sv}:")
        print(f"   dX = {dx:14.3f} m")
        print(f"   dY = {dy:14.3f} m")
        print(f"   dZ = {dz:14.3f} m\n")
    print("----------------------------------------------------------------------------------------------")
    
    


    
    # Task 3: Transform receiver coordinates from geodetic to cartesian
    X0, Y0, Z0 = geodetic_to_cartesian(receiver_lat0, receiver_lon0, receiver_h0)
    print("\n=== Approximate receiver cartesian coordinates from given geodetic ===\n")
    print(f"  X0 = {X0:,.3f} m")
    print(f"  Y0 = {Y0:,.3f} m")
    print(f"  Z0 = {Z0:,.3f} m\n")
    print("----------------------------------------------------------------------------------------------")





    # Task 4: Estimate receiver position in Cartesian coordinates using Least Squares
    A, dL, (Xi, Yi, Zi), dT, iteration, Qx_last, PDOP = least_squares_position(satellites, sat_positions, [X0, Y0, Z0], P_L1, DT, DION, DTROP, c)
    print("\n=== Observation equations and the design matrix ===\n")
    print("Observation vector dL:")
    print(dL)
    print("\nDesign matrix A:")
    print(A)
    print("\n=== Estimated receiver position by least squares method in cartesian coordinates ===\n")
    print(f"  Xi = {Xi:,.3f} m")
    print(f"  Yi = {Yi:,.3f} m")
    print(f"  Zi = {Zi:,.3f} m")
    print(f"  Number of iterations: {iteration}\n")
    print("----------------------------------------------------------------------------------------------")





    # Task 5: Compute the positional dilution of precision (PDOP)
    print("\n=== Positional Dilution of Precision (PDOP) ===\n")
    print("Qx matrix (cofactor matrix):")
    print(Qx_last)
    print(f"  PDOP = {PDOP:.6f}\n")
    print("----------------------------------------------------------------------------------------------")





    # Task 6: Compure the receiver position in geodetic coordinates
    lat, lon, h = cartesian_to_geodetic(Xi, Yi, Zi)
    print("\n=== Estimated receiver position in geodetic coordinates ===\n")
    print(f"  Latitude  = {lat:.6f} degrees")
    print(f"  Longitude = {lon:.6f} degrees")
    print(f"  Height    = {h:.3f} m\n")
    print("----------------------------------------------------------------------------------------------")





    # Task 7: Estimate receiver clock error
    print("\n=== Estimated receiver clock error ===\n")
    print(f"  dT = {dT:.12f} seconds\n")
    print("----------------------------------------------------------------------------------------------")