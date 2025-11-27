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

# where the stationary distributions are stored
path_stat_distns = "within_game_stationary_distribution.parquet"

# where the action-pair payoffs are stored 
path_actions_pays = "actions_payoffs.parquet"

# where to store results
path_out = "within_game_stationary_payoffs.parquet"


# read in the payoff to players in each action pair 
# ---

table = pq.read_table(path_actions_pays, columns = ["actions_ID", "payoffs"])
ID_2_pays = dict(zip(table["actions_ID"].to_pylist(), table["payoffs"].to_pylist()))


# read in the stationary distributions 
# ---

table = pq.read_table(
        path_stat_distns,
        columns = [
            "player_1_fsauthor_strategy", 
            "player_2_fsauthor_strategy", 
            "player_1_coauthor_strategy", 
            "player_2_coauthor_strategy", 
            "stationary_distn_numerators", 
            "stationary_distn_denominators"
        ]
)


# calculate the expected payoffs to the focal and other at the stationary distributions 
# ---

stat_distn_numers_denoms = list(zip(
    table["stationary_distn_numerators"].to_pylist(), 
    table["stationary_distn_denominators"].to_pylist()
))

p1_pays = list()
p2_pays = list()
for stat_distn_numers, stat_distn_denoms in stat_distn_numers_denoms:
    p1_pay = sum(
            ID_2_pays[ID][0] * stat_distn_numer / stat_distn_denom 
            for ID, (stat_distn_numer, stat_distn_denom) in enumerate(zip(stat_distn_numers, stat_distn_denoms))
    )
    p2_pay = sum(
            ID_2_pays[ID][1] * stat_distn_numer / stat_distn_denom 
            for ID, (stat_distn_numer, stat_distn_denom) in enumerate(zip(stat_distn_numers, stat_distn_denoms))
    )
    p1_pays.append(p1_pay)
    p2_pays.append(p2_pay)


# write to parquet
# ---

# build a new table from scratch
table = pa.table({
  **table.to_pydict(),
  "player_1_payoff": p1_pays,
  "player_2_payoff": p2_pays,
})

pq.write_table(table, path_out)
