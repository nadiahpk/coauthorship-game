# Plot an example of the deterministic within-game transitions 

import os
from pathlib import Path
import pandas as pd

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
coauthor_strats = ["pavlov_d", "pavlov_d"]
fsauthor_strats = ["never", "never"]

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


# double check the regime we're in by printing to the screen
# ---

# print to screen the alignment between the two players' topics
topics = mf.topics_fnc(params)
alignments = mf.alignments_fnc(params, preferences, topics)
L = alignments[0][1]
print(f"Topic alignment L = {L}")

# first-author switching gains

# if the other is not co-authoring
ID = actions_2_ID[((0, 0), (0, 0))]
fsauthor_gain_positive = action_gains_positive[ID][0][0]
if fsauthor_gain_positive:
    print("It is in the focal's self-interest to independently first-author")
else:
    print("It is NOT in the focal's self-interest to independently first-author")

# if the other is co-authoring
ID = actions_2_ID[((0, 0), (1, 0))]
fsauthor_gain_positive = action_gains_positive[ID][0][0]
if fsauthor_gain_positive:
    print("It is in the focal's self-interest to first-author if the non-focal is coauthoring")
else:
    print("It is NOT in the focal's self-interest to first-author even if the non-focal is coauthoring")

# co-author switching gain
ID = actions_2_ID[((0, 0), (0, 1))]
coauthor_gain_positive = action_gains_positive[ID][0][1]
if coauthor_gain_positive:
    print("It is in the focal's self-interest to co-author")
else:
    print("It is NOT in the focal's self-interest to co-author")




# for each possible n-player action-state, calculate the deterministic next state
# ---

# takes into account both the switching gain and the focal's authorship strategies
deterministic_transitions = mf.calc_deterministic_transitions(
    params, actions_2_ID, action_gains_positive, fsauthor_strats, coauthor_strats
)



# plot to check
# ---

# create the networkx graph
G = nx.DiGraph(np.array(deterministic_transitions))

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


AG = vf.create_attributes_deterministic_graph(G, ID_2_actions, ID_2_fillcolor)
vf.print_attributes_deterministic_graph(AG, "example_within_game_transitions")
