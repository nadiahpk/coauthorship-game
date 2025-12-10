# create a simple example to explain how the stationary distribution
# is solved analytically
#

"""
# decided too complicated
# I'm going to do a 4 state system where the deterministic graph is
#   A -> C -> C, B -> D -> D
# and the stochastic graph has
#   C -> A: e^2
#   C -> B: (1 - e) e
#   D -> A: (1 - e) e
#   D -> B: e^2
# construct P
P = [
    [   0,    0,      1,      0     ],
    [   0,    0,      0,      1     ],
    [ eps**2, eps * (1 - eps),    (1 - eps)**2,    0     ],
    [ eps * (1 - eps),  eps**2,    0,    D-D     ],
]
"""

import sympy as sp
import numpy as np


# construct symbolic P
# ---

# epsilon is the action-error probability
eps = sp.symbols("eps")
P = np.array(
    [
        [1 - eps, eps],
        [eps, 1 - eps],
    ]
)

max_pwr = 1

# find stationary distribution
# ---

nbr_states = P.shape[0]

# matrix of epsilon coefficients, k_ij
# where rows i correspond to the state
# and columns j correspond to the power of epsiol
coeffs = [
    [sp.symbols(f"k{state_idx}{pwr}") for pwr in range(max_pwr + 1)]
    for state_idx in range(nbr_states)
]

# the stationary distribution expressed as a polynomial in epsilon
# v_i = k_i0 eps^0 + k_i1 eps^1 + ...
v = np.array(
    [
        sum(coeffs[state_idx][pwr] * eps**pwr for pwr in range(max_pwr + 1))
        for state_idx in range(nbr_states)
    ]
)

# at the stationary distribution, v = vP
rhsV = list(v @ P)

# At the stationary distribution, v = rhsV.
# As the error eps -> 0, the stationary distribution approaches
# the epsilon-order 0 coefficients, i.e.,
#   v -> (k_10, k_20, k_30, ...)

# solve by matching epsilon-power terms

# first, from normalisation of the stationary distribution,
# we always have: 1 = k_10 + k_20 + k_30 + ...
pwr_2_eq0s = {0: [1 - sum(coeffs[state_idx][0] for state_idx in range(nbr_states))]}

# then, from v = vP, each state has an equation involving various
# epsilon-order terms:
#   k_i0 + k_i1 eps + ... = [term]_i0 + [term]_i1 eps + ...
for state_idx, rhs in enumerate(rhsV):
    pwrs_terms = rhs.as_poly(eps).all_terms()
    for pwr_tuple, term in pwrs_terms:
        pwr = pwr_tuple[0]
        if pwr <= max_pwr:
            eq0 = term - coeffs[state_idx][pwr]
            if eq0 != 0:  # exclude not-useful 0 = 0 equations
                pwr_2_eq0s.setdefault(pwr, []).append(eq0)

# we only want to solve the values of k_10, k_20, ...
wants = [coeffs[state_idx][0] for state_idx in range(nbr_states)]

# so we match epsilon terms, increasing the epsilon power
# incrementally, until we have taken into account enough rare
# sequences of errors that we can obtain the solutions we want
stationary_distn = list()
eq0s = list()
for pwr_level in range(max_pwr + 1):
    # add to our system of equations
    # coefficient-matching at current power level
    eq0s += pwr_2_eq0s[pwr_level]

    # solve system of equations for epsilon coefficients
    free_coeffs = set.union(*[eq0.free_symbols for eq0 in eq0s])
    soln = sp.solve(eq0s, free_coeffs)  # returns [] if can't

    # check solution
    if all(want in soln for want in wants):
        # got something for each coeff we want
        stationary_distn_temp = [soln[want] for want in wants]
        if all([not propn.free_symbols for propn in stationary_distn_temp]):
            # each something we got had no unknown variables
            stationary_distn = stationary_distn_temp
            break
