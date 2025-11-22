import numpy as np
import itertools

from LS_computer import (
    least_squares_position_and_dd_ambiguities_float_solution,
    least_squares_position_with_fixed_dd_ambiguities,
)

def full_ambiguity_search(A_known_pos_cartesian, B_approx_pos_cartesian,
                          SAT_POS, L1, lambda_L1,
                          ref_satellite, other_satellites, epochs,
                          k_sigma=3.0):
    """
    Scenario (c): full ambiguity search using ±k_sigma * sigma_N
    around the float ambiguities, testing all integer combinations.
    """

    # 1. Float solution from Task 2
    A, dL, P, x_hat, C_x, dX_B_float, dd_ambiguities_float, B_pos_float = \
        least_squares_position_and_dd_ambiguities_float_solution(
            A_known_pos_cartesian,
            B_approx_pos_cartesian,
            SAT_POS,
            L1,
            lambda_L1,
            ref_satellite,
            other_satellites,
            epochs,
        )

    # 2. Standard deviations of ambiguities (last 4 unknowns)
    var_N = np.diag(C_x)[3:]      # ambiguities are components 3..6
    sigma_N = np.sqrt(var_N)

    # 3. Build candidate integer list for each ambiguity
    candidate_lists = []
    for i, N_float in enumerate(dd_ambiguities_float):
        sig = sigma_N[i]

        if sig == 0:
            candidate_lists.append([int(round(N_float))])
            continue

        low = N_float - k_sigma * sig
        high = N_float + k_sigma * sig

        cand_min = int(np.floor(low))
        cand_max = int(np.ceil(high))

        if cand_max < cand_min:
            # fallback: nearest integer only
            candidate_lists.append([int(round(N_float))])
        else:
            candidates = list(range(cand_min, cand_max + 1))
            candidate_lists.append(candidates)

    # 4. All combinations of integer ambiguities
    all_combinations = list(itertools.product(*candidate_lists))

    best_SSR = None
    second_best_SSR = None
    best_solution = None
    second_best_solution = None

    # 5. Evaluate each combination
    for combo in all_combinations:
        N_fix = np.array(combo, dtype=float)

        _, dL_pos, x_hat_pos, C_x_pos, B_pos_fixed, v, SSR = \
            least_squares_position_with_fixed_dd_ambiguities(
                B_approx_pos_cartesian,
                lambda_L1,
                other_satellites,
                epochs,
                N_fix,
                dL,
                A,
                P,
            )

        if best_SSR is None or SSR < best_SSR:
            # move current best to second best
            second_best_SSR = best_SSR
            second_best_solution = best_solution

            # update best
            best_SSR = SSR
            best_solution = {
                "N_fix": N_fix,
                "B_pos": B_pos_fixed,
                "C_pos": C_x_pos,
                "SSR": SSR,
                "v": v,
                "x_hat_pos": x_hat_pos,
            }
        elif second_best_SSR is None or SSR < second_best_SSR:
            second_best_SSR = SSR
            second_best_solution = {
                "N_fix": N_fix,
                "B_pos": B_pos_fixed,
                "C_pos": C_x_pos,
                "SSR": SSR,
                "v": v,
                "x_hat_pos": x_hat_pos,
            }

    # 6. Ratio test
    if best_SSR is not None and second_best_SSR is not None:
        ratio = second_best_SSR / best_SSR
    else:
        ratio = None

    return {
        "dd_ambiguities_float": dd_ambiguities_float,
        "sigma_N": sigma_N,
        "candidate_lists": candidate_lists,
        "all_combinations_count": len(all_combinations),
        "best": best_solution,
        "second_best": second_best_solution,
        "best_SSR": best_SSR,
        "second_best_SSR": second_best_SSR,
        "ratio_SSR": ratio,
        "A": A,
        "dL": dL,
        "P": P,
        "C_x_float": C_x,
    }
