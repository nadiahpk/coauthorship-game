# Load the deterministic transitions and calculate the stationary distribution as action error approaches zero

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# import numpy as np
import sympy as sp

import model_fncs as mf


# parameters
# ---

# nbr players
n = 2

# where the deterministic transitions are stored
path_in = "within_game_deterministic_transitions.parquet"

# where the count of errors for each transitions are stored
fname_error_count = Path(os.environ.get('RESULTS', "")) / "deterministic" / f"transitions_errors_nbr_players_{n}.parquet"

# where to save the results from this script
path_out = "within_game_stationary_distribution.parquet" 


# get the dictionary mapping action IDs to actions 
# ---

# ID_2_actions = mf.get_ID_2_actions(n)
# ID_2_actions_flattened = 

# Not necessary, just need to know how many there are
nbr_actions = (n * n)**2


# calculate the stationary distribution
# ---

# load in deterministic transitions
table = pq.read_table(path_in)

# place to store the numerator and denominator of stationary distributions
stationary_distn_numersV = list()
stationary_distn_denomsV = list()

# TODO NOTE to future self: using the epsilon error to find the 
# stationary distribution is only necessary if the within-game 
# deterministic dynamics has more than one attractor. It may be
# more efficient to count the number of attractors first.

# loop through each scenario, finding stationary distribution
for scenario_nbr, row in enumerate(table.to_pylist()):

    # get list of transitions as native Python types
    transitions = row["deterministic_transitions_list"]

    # create a dictionary of the deterministic transitions from predecessor (key)
    # to successor (value)
    pre_2_det = dict(transitions)

    # get the transition matrix with error terms included (symbolic entries a fnc of eps)
    P, eps = mf.symbolic_transitions_with_errors(pre_2_det, fname_error_count)

    # find the stationary distribution as eps -> 0
    # max_pwr = 2 # maximimum power level of terms to include, defaulted
    stationary_distn_sp, pwr_level = mf.find_stationary_distribution(P, eps)

    # convert the sumpy fractions to numerator-denominator integer pairs so we can store them safely
    stationary_distn_numersV.append([int(sp.numer(v)) for v in stationary_distn_sp])
    stationary_distn_denomsV.append([int(sp.denom(v)) for v in stationary_distn_sp])


# write to parquet
# ---

# columns are: player_1_fsauthor_strategy, player_2_fsauthor_strategy, player_1_coauthor_strategy, player_2_coauthor_strategy, deterministic_transitions_list
table = pa.table({
    "player_1_fsauthor_strategy": table["player_1_fsauthor_strategy"].to_pylist(),
    "player_2_fsauthor_strategy": table["player_2_fsauthor_strategy"].to_pylist(),
    "player_1_coauthor_strategy": table["player_1_coauthor_strategy"].to_pylist(),
    "player_2_coauthor_strategy": table["player_2_coauthor_strategy"].to_pylist(),
    "stationary_distn_numerators": stationary_distn_numersV,
    "stationary_distn_denominators": stationary_distn_denomsV,
})

pq.write_table(table, path_out)

