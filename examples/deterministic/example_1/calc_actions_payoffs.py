# For each possible pair of actions, calculate the payoff that each 
# player will receive in that round

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import itertools as it

import model_fncs as mf


# parameters
# ---

# where to save results
path_out = "actions_payoffs.parquet" 

# game parameters
n = 2  # number of players
b = 2  # maximum benefit of a paper
c_f = 0.9  # cost of first-authoring
c_c = 0.8  # cost of co-authoring

# topic preferences
preferences = np.array([[0], [2]])
assert len(preferences) == n

# strategies used by each player
fsauthor_strat = "never" # just do one for now
valid_coauthor_strats = ["pavlov_c", "pavlov_d", "all_c", "all_d"]

# store in the parameter-values function for this run
params = {
    "nbr_players": n,
    "preferences": preferences,
    "type_of_good": "excludable",
    "gamma_rule": "constant_1",
    "first_author_weight": 1,  # benefit of papers
    "max_paper_benefit": b,
    "distance_measure": "euclidean",  # alignment between preference and topic
    "authoring_cost_rule": "simple",  # authoring costs
    "first_author_cost": c_f,
    "coauthor_cost": c_c,
    "topic_choice_rule": "exact_preference",  # topic-choice
}


# index every possible n-player action state
# ---

ID_2_actions = mf.get_ID_2_actions(n)
IDs = sorted(ID_2_actions.keys()) # distrust dictionary ordering


# for each possible pair of actions, calculate payoff to each player
# ---

topics = mf.topics_fnc(params)
paysV = [mf.calc_pays(params, ID_2_actions[ID], topics) for ID in IDs]


# save to parquet file 
# ---

actionsV = [ID_2_actions[ID] for ID in IDs]
table = pa.table({
    "actions_ID": IDs,
    "actions": actionsV,
    "payoffs": paysV,
})

pq.write_table(table, path_out)
