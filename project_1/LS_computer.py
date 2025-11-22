import numpy as np


def least_squares_position(satellites, sat_positions, approx_receiver_pos, P_L1, DT, DION, DTROP, c, max_iter: int = 10, tol_pos: float = 1e-4, tol_clk: float = 1e-12):
    Xi, Yi, Zi = approx_receiver_pos[0], approx_receiver_pos[1], approx_receiver_pos[2]
    dT = 0.0 # initial receiver clock bias
    Qx_last = None

    for iteration in range(max_iter):
        # Build design matrix A and observation vector dL
        A_rows = []
        dL_rows = []

        for sv in satellites:
            Xj, Yj, Zj = sat_positions[sv]
            dx = Xj - Xi
            dy = Yj - Yi
            dz = Zj - Zi
            rho = np.sqrt(dx**2 + dy**2 + dz**2)
            aX = -dx / rho
            aY = -dy / rho
            aZ = -dz / rho
            dL = P_L1[sv] - rho - c*DT[sv] - DION[sv] - DTROP[sv]

            A_rows.append([aX, aY, aZ, -c])
            dL_rows.append([dL])

        A = np.array(A_rows)    # shape (7,4)
        dL = np.array(dL_rows)  # shape (7,1)

        # Normal equation
        N = A.T @ A
        n = A.T @ dL
        dX = np.linalg.solve(N, n)
        dXi, dYi, dZi, dT_i = dX.flatten()

        # Update receiver position and clock bias
        Xi_new = Xi + dXi
        Yi_new = Yi + dYi
        Zi_new = Zi + dZi
        dT_new = dT_i

        # Save cofactor matrix for PDOP
        Qx_last = np.linalg.inv(N)

        # Check convergence
        if (abs(dXi) < tol_pos and abs(dYi) < tol_pos and abs(dZi) < tol_pos and abs(dT_i) < tol_clk):
            Xi, Yi, Zi, dT = Xi_new, Yi_new, Zi_new, dT_new
            break
        Xi, Yi, Zi, dT = Xi_new, Yi_new, Zi_new, dT_new

    # Compute PDOP
    PDOP = np.sqrt(Qx_last[0,0] + Qx_last[1,1] + Qx_last[2,2])

    return A, dL, (Xi, Yi, Zi), dT, iteration, Qx_last, PDOP
    


        



        
  

