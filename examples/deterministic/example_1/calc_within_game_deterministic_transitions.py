# Write the deterministic, within-game transition matrices to a file

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
path_out = "within_game_deterministic_transitions.parquet" 

# game parmaeters
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
actions_2_ID = {actions: ID for ID, actions in ID_2_actions.items()}


# calculations that are agnostic about authorship strategy
# ---

# the simplifying assumptions of this model mean gammas and alignments are constant
gammas = [[mf.gamma_fnc(params, i, j) for j in range(n)] for i in range(n)]
topics = mf.topics_fnc(params)
alignments = mf.alignments_fnc(params, preferences, topics)

# gains that are positive mean that, for authorship in that instance (first- or co-authorship),
# if the focal player unilaterally switches from not-author to author (a_{i,j} switches from 0 to 1),
# then the direct authorship payoffs exceed the costs
#
# in other words, a player should always author in this situation
action_gains_positive = mf.calc_action_gains_positive(params, ID_2_actions, gammas, alignments)


# loop through each combination of player strategies
# ---

player_1_coauthor_strats, player_2_coauthor_strats = zip(*it.combinations_with_replacement(valid_coauthor_strats, 2))
len_rows = len(player_1_coauthor_strats)
player_x_fsauthor_strats = [fsauthor_strat] * len_rows
deterministic_transitions_lists = list() 

fsauthor_strats = [fsauthor_strat, fsauthor_strat]

for player_1_coauthor_strat, player_2_coauthor_strat in zip(player_1_coauthor_strats, player_2_coauthor_strats):
    coauthor_strats = [player_1_coauthor_strat, player_2_coauthor_strat]

    # for each possible n-player action-state, calculate the deterministic next state

    # takes into account both the switching gain and the focal's authorship strategies
    deterministic_transitions_matrix = mf.calc_deterministic_transitions(
        params, actions_2_ID, action_gains_positive, fsauthor_strats, coauthor_strats
    )

    # turn it into a list of tuples (from_ID, to_ID) where the ID is the action ID from actions_2_ID
    deterministic_transitions_list = [[row_idx, row.index(1)] for row_idx, row in enumerate(deterministic_transitions_matrix)]
    
    # save 
    deterministic_transitions_lists.append(deterministic_transitions_list)


# write to parquet
# ---

# columns are: player_1_fsauthor_strategy, player_2_fsauthor_strategy, player_1_coauthor_strategy, player_2_coauthor_strategy, deterministic_transitions_list
table = pa.table({
    "player_1_fsauthor_strategy": player_x_fsauthor_strats,
    "player_2_fsauthor_strategy": player_x_fsauthor_strats,
    "player_1_coauthor_strategy": player_1_coauthor_strats,
    "player_2_coauthor_strategy": player_2_coauthor_strats,
    "deterministic_transitions_list": deterministic_transitions_lists,
})

pq.write_table(table, path_out)
