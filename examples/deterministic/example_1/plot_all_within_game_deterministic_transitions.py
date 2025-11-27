# Plot an example of the deterministic within-game transitions 

import os
from pathlib import Path
import pandas as pd
import itertools as it
import numpy as np
import networkx as nx

import model_fncs as mf
import vis_fncs as vf

# parameters
# ---

# game parmaeters
n = 2  # number of players
b = 2  # maximum benefit of a paper
c_f = 0.9  # cost of first-authoring
c_c = 0.8  # cost of co-authoring

# topic preferences
preferences = np.array([[0], [2]])
assert len(preferences) == n

# strategies used by each player
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


# for each combination of player strategies, plot deterministic graph
# ---

# define fill colours for each action-state
fsauthor_2_fillcolor = {
    (0, 0): "#ffcccc",
    (0, 1): "#eeeebb",
    (1, 0): "#ccddaa",
    (1, 1): "#bbccee",
}
ID_2_fillcolor = {
    ID: fsauthor_2_fillcolor[tuple([actions[i][i] for i in range(n)])]
    for ID, actions in ID_2_actions.items()
}

# ready list for looping over
player_1_coauthor_strats, player_2_coauthor_strats = zip(*it.combinations_with_replacement(valid_coauthor_strats, 2))
len_rows = len(player_1_coauthor_strats)
player_x_fsauthor_strats = [fsauthor_strat] * len_rows

fsauthor_strats = [fsauthor_strat, fsauthor_strat]

for player_1_coauthor_strat, player_2_coauthor_strat in zip(player_1_coauthor_strats, player_2_coauthor_strats):
    coauthor_strats = [player_1_coauthor_strat, player_2_coauthor_strat]

    # for each possible n-player action-state, calculate the deterministic next state

    # takes into account both the switching gain and the focal's authorship strategies
    deterministic_transitions = mf.calc_deterministic_transitions(
        params, actions_2_ID, action_gains_positive, fsauthor_strats, coauthor_strats
    )

    # save plot of graph

    # create the networkx graph
    G = nx.DiGraph(np.array(deterministic_transitions))

    # get the graph with nice attributes
    AG = vf.create_attributes_deterministic_graph(G, ID_2_actions, ID_2_fillcolor)

    fname = f"within_game_deterministic_transitions_{player_1_coauthor_strat}_Vs_{player_2_coauthor_strat}"
    vf.print_attributes_deterministic_graph(AG, fname)
