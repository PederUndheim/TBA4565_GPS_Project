import numpy as np

# Weight matrix P for 5 satellites -> 4 DD per epoch, two epochs -> 8x8 matrix
def weight_matrix(sigma=0.005):
    J = 5*np.eye(4) - np.ones((4, 4))  # diag=4, offdiag=-1
    scale = 1.0 / (2.0 * sigma**2 * 5.0)
    P = scale * np.block([[J, np.zeros((4, 4))], [np.zeros((4, 4)), J]])
    return P

# Compute geometric distance between satellite and receiver
def _rho(sat_pos, rec_pos):
    return np.linalg.norm(sat_pos - rec_pos)


# Least squares estimation of receiver position and double difference ambiguities (float solution)
def least_squares_position_and_dd_ambiguities_float_solution(
        A_known_pos, B_approx_pos, SAT_POS, L1, lambda_L1, ref_satellite, other_satellites, epochs, tol=1e-6, max_iter=10):
    
    P = weight_matrix()
    B_curr = B_approx_pos.copy() # Initial guess for B position that will be updated iteratively   

    for iteration in range(max_iter):

        n_dd_per_epoch = len(other_satellites)      # 4
        n_epochs = len(epochs)                      # 2
        n_obs = n_dd_per_epoch * n_epochs           # 8
        n_unknowns = 3 + n_dd_per_epoch             # 7

        # Map each double difference (ref_sat, other_sat) to ambiguity index 0 to 3
        amb_index = {sat: idx for idx, sat in enumerate(other_satellites)}

        # Design observation vector deltaL and design matrix A
        dL = np.zeros(n_obs)
        A = np.zeros((n_obs, n_unknowns))

        row = 0
        for epoch in epochs:
            for satellite_j in other_satellites:

                # Carrier phase double difference (Φ_AB_ij(t) * λ)
                phi_A_i = L1["A"][epoch][ref_satellite] * lambda_L1
                phi_A_j = L1["A"][epoch][satellite_j] * lambda_L1
                phi_B_i = L1["B"][epoch][ref_satellite] * lambda_L1
                phi_B_j = L1["B"][epoch][satellite_j] * lambda_L1
                Phi_AB_ij = phi_B_j - phi_B_i - phi_A_j + phi_A_i

                # Geometric distances (approximate positions for B)
                pos_i_A = np.array(SAT_POS["A"][epoch][ref_satellite])
                pos_j_A = np.array(SAT_POS["A"][epoch][satellite_j])
                pos_i_B = np.array(SAT_POS["B"][epoch][ref_satellite])
                pos_j_B = np.array(SAT_POS["B"][epoch][satellite_j])

                rho_A_i = _rho(pos_i_A, A_known_pos)
                rho_A_j = _rho(pos_j_A, A_known_pos)
                rho_B0_i = _rho(pos_i_B, B_curr)
                rho_B0_j = _rho(pos_j_B, B_curr)

                dL[row] = Phi_AB_ij - rho_B0_j + rho_B0_i + rho_A_j - rho_A_i


                # Partial derivatives for receiver position
                dBj = pos_j_B - B_curr
                dBi = pos_i_B - B_curr

                aX_B = -(dBj[0] / rho_B0_j) + (dBi[0] / rho_B0_i)
                aY_B = -(dBj[1] / rho_B0_j) + (dBi[1] / rho_B0_i)
                aZ_B = -(dBj[2] / rho_B0_j) + (dBi[2] / rho_B0_i)

                A[row, 0] = aX_B
                A[row, 1] = aY_B
                A[row, 2] = aZ_B

                col_N = amb_index[satellite_j] + 3
                A[row, col_N] = lambda_L1

                row += 1

        # Least squares solution
        N = A.T @ P @ A
        n = A.T @ P @ dL
        x_hat = np.linalg.solve(N, n)

        # Variance-covariance matrix of the estimated parameters
        C_x = np.linalg.inv(N)

        # Prepare output
        dX_B = x_hat[0:3]               # Position correction for B
        dd_ambiguities = x_hat[3:]      # Double difference ambiguities

        # Update B
        B_new = B_curr + dX_B

        # Check convergence
        if np.max(np.abs(dX_B)) < tol:
            B_curr = B_new
            break

        B_curr = B_new

    return A, dL, P, x_hat, C_x, dX_B, dd_ambiguities, B_curr



# Least squares estimation of receiver position with fixed integer double difference ambiguities
def least_squares_position_with_fixed_dd_ambiguities(
        B_approx_pos, lambda_L1, other_satellites, epochs,
        N_fix, dL, A, P):
        
    # 1. Make new observation vector with fixed ambiguities (ΔL* = ΔL - λ*N_fix)
    dL_pos = np.zeros_like(dL)
    idx = 0
    for epoch in epochs:
        for j_idx, sat_j in enumerate(other_satellites):
            dL_pos[idx] = dL[idx] - lambda_L1 * N_fix[j_idx]
            idx += 1

    # 2. Design matrix for position only (3 unknowns)
    A_pos = A[:, :3]  # 8x3

    # 3. LS solution
    N_mat = A_pos.T @ P @ A_pos
    n_vec = A_pos.T @ P @ dL_pos
    x_hat_pos = np.linalg.solve(N_mat, n_vec)
    C_x_pos = np.linalg.inv(N_mat)

    B_pos_cartesian_fixed = B_approx_pos + x_hat_pos

    # 4. Residuals and weighted SSR
    v = dL_pos - A_pos @ x_hat_pos          # 8×1
    SSR = float(v.T @ P @ v)                # scalar

    return A_pos, dL_pos, x_hat_pos, C_x_pos, B_pos_cartesian_fixed, v, SSR