import numpy as np

from coordinate_transformer import geodetic_to_cartesian, cartesian_to_geodetic
from base_rover_data import SAT_POS, L1
from LS_computer import least_squares_position_and_dd_ambiguities_float_solution, least_squares_position_with_fixed_dd_ambiguities
from full_ambiguity_search import full_ambiguity_search
# SAT_POS = {"A": {"t1": {154: (-26916298.03,  -2738678.66, -11996368.48),....
# L1 = {"A": {"t1": {154: 143588831.82, ....

L1_freq = 1575.42e6        # L1 frequency in Hz
c = 299792458              # Speed of light in m/s
lambda_L1 = c / L1_freq    # Wavelength of L1 in meters 

# Epoch times (seconds)
epochs = ["t1", "t2"]
t1 = 172800
t2 = 175020

# List of satellites
satellites = [154, 155, 159, 174, 181]

# Known and approximate geodetic receiver positions
A_known_pos_geodetic = [-32.003884648, 115.894802001, 23.983]  # [lat (deg), lon (deg), h (m)]
B_approx_pos_geodetic = [-31.9, 115.75, 50.0]                  # [lat (deg), lon (deg), h (m)]



if __name__ == "__main__":

    # Task 1: Transform receiver coordinates to Cartesian coordinates
    A_known_pos_cartesian = geodetic_to_cartesian(*A_known_pos_geodetic)
    B_approx_pos_cartesian = geodetic_to_cartesian(*B_approx_pos_geodetic)
    
    print("----------------------------------------------------------------------------------------------")
    print("\n=== Receiver Cartesian coordinates ===\n")
    print(f"Base station (A):")
    print(f"   X = {A_known_pos_cartesian[0]:.3f} m")
    print(f"   Y = {A_known_pos_cartesian[1]:.3f} m")
    print(f"   Z = {A_known_pos_cartesian[2]:.3f} m\n")
    print(f"Rover station (B) approximate:")
    print(f"   X = {B_approx_pos_cartesian[0]:.3f} m")
    print(f"   Y = {B_approx_pos_cartesian[1]:.3f} m")
    print(f"   Z = {B_approx_pos_cartesian[2]:.3f} m\n")
    print("----------------------------------------------------------------------------------------------\n") 




    # Task 2: Estimate receiver position using double difference, least squares, carrier phase observations (float solution) + variance-covariance matrix
    A, dL, P, x_hat, C_x, dX_B, dd_ambiguities, B_pos_cartesian = least_squares_position_and_dd_ambiguities_float_solution(
        A_known_pos_cartesian, B_approx_pos_cartesian, SAT_POS, L1, lambda_L1, satellites[0], satellites[1:], epochs)
    print("\n=== Least Squares Estimation Results ===\n")
    print("Design Matrix (A):")
    print(A)
    print("\nObservation Vector (dL):")
    print(dL)
    print("\nWeight Matrix (P):")
    print(P)
    print("\nEstimated Parameter Corrections (x̂):")
    print(x_hat)
    print("\nVariance-Covariance Matrix of Estimated Parameters (C_x):")
    print(C_x)
    print("\nEstimated Position Corrections for Rover Station B (dX_B):")
    print(dX_B)
    print("\nEstimated Double Difference Ambiguities (dd_ambiguities):")
    print(dd_ambiguities)
    print("\nEstimated Rover Station B Cartesian Coordinates:")
    print(f"   X = {B_pos_cartesian[0]:.3f} m")
    print(f"   Y = {B_pos_cartesian[1]:.3f} m")
    print(f"   Z = {B_pos_cartesian[2]:.3f} m\n")
    print("----------------------------------------------------------------------------------------------\n")




    # Task 3: Fix the ambiguities to integer values and re-estimate the rover position
    print("\n=== Rover Position Estimation with different searching procedures for ambiguity fix ===\n")

    # a) Do nothing with the ambiguities (keep float solution)
    N_fix_a = dd_ambiguities.copy()
    A_pos, dL_pos, x_hat_pos, C_x_pos_a, B_pos_cartesian_fixed_a, v_a, SSR_a = \
        least_squares_position_with_fixed_dd_ambiguities(
            B_pos_cartesian, lambda_L1, satellites[1:], epochs,
            N_fix_a, dL, A, P
        )
    print("a) Keeping float solution:")
    print(f"--> N values: {N_fix_a}")
    print(f"--> SSR_a (v^T P v): {SSR_a:.4f}")
    print("Estimated Rover Station B Cartesian Coordinates:")
    print(f"   X = {B_pos_cartesian_fixed_a[0]:.3f} m")
    print(f"   Y = {B_pos_cartesian_fixed_a[1]:.3f} m")
    print(f"   Z = {B_pos_cartesian_fixed_a[2]:.3f} m\n")

    # b) Rounding to nearest integer
    N_fix_b = np.round(dd_ambiguities)
    A_pos, dL_pos, x_hat_pos, C_x_pos_b, B_pos_cartesian_fixed_b, v_b, SSR_b = \
        least_squares_position_with_fixed_dd_ambiguities(
            B_pos_cartesian, lambda_L1, satellites[1:], epochs,
            N_fix_b, dL, A, P
        )
    print("b) Rounding to nearest integer:")
    print(f"--> N values: {N_fix_b}")
    print(f"--> SSR_b (v^T P v): {SSR_b:.4f}")
    print("Estimated Rover Station B Cartesian Coordinates:")
    print(f"   X = {B_pos_cartesian_fixed_b[0]:.3f} m")
    print(f"   Y = {B_pos_cartesian_fixed_b[1]:.3f} m")
    print(f"   Z = {B_pos_cartesian_fixed_b[2]:.3f} m\n")

    # c) Full search based on standard deviations (±3σ window)
    result_c = full_ambiguity_search(
        np.array(A_known_pos_cartesian),
        np.array(B_pos_cartesian),
        SAT_POS,
        L1,
        lambda_L1,
        satellites[0],       # ref satellite
        satellites[1:],      # other satellites
        epochs,
        k_sigma=3.0,
    )

    best_c = result_c["best"]
    second_best_c = result_c["second_best"]
    N_fix_c = best_c["N_fix"]
    B_pos_cartesian_fixed_c = best_c["B_pos"]
    C_x_pos_c = best_c["C_pos"]
    SSR_c = best_c["SSR"]
    second_best_SSR_c = second_best_c["SSR"]

    print("c) Full ambiguity search (±3σ):")
    print(f"--> float ambiguities: {result_c['dd_ambiguities_float']}")
    print(f"--> σ_N (std devs):    {result_c['sigma_N']}")
    print(f"--> candidate lists:   {result_c['candidate_lists']}")
    print(f"--> combinations tested: {result_c['all_combinations_count']}")
    print(f"--> best N_fix_c:      {N_fix_c}")
    print(f"--> SSR_c (v^T P v):   {SSR_c:.4f}")
    print(f"--> second best N_fix_c:    {second_best_c['N_fix']}")
    print(f"--> second best SSR:   {second_best_SSR_c:.4f}")
    print(f"--> SSR ratio (2nd/1st): {result_c['ratio_SSR']}\n")
    print("Estimated Rover Station B Cartesian Coordinates (scenario c):")
    print(f"   X = {B_pos_cartesian_fixed_c[0]:.3f} m")
    print(f"   Y = {B_pos_cartesian_fixed_c[1]:.3f} m")
    print(f"   Z = {B_pos_cartesian_fixed_c[2]:.3f} m\n")
    print("----------------------------------------------------------------------------------------------\n")




    # Task 4: Transform final rover position back to geodetic coordinates
    lat_B, lon_B, h_B = cartesian_to_geodetic(*B_pos_cartesian_fixed_c)
    print("=== Final Rover Station B Geodetic Coordinates ===\n")
    print(f"   Latitude  = {lat_B:.9f} degrees")
    print(f"   Longitude = {lon_B:.9f} degrees")
    print(f"   Height    = {h_B:.3f} m\n")



