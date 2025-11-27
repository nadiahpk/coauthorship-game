# Write the full in-game deterministic transition matrix,
#
# I've decided instead co-authorship memory should only update
# when the co-authorship actually happens, i.e., the first-author
# published their paper

import numpy as np
import networkx as nx

import examples_fncs_3 as mf


# --------------------------------

# varying parameters
# ---

# NOTE lazy commenting out scenarios here

# prefix_out = "eg_7_pavlov_c"
# coauthor_strats = ["pavlov_c", "pavlov_c"]

# prefix_out = "eg_7_all_d_always"
# coauthor_strats = ["all_d", "all_d"]
# fsauthor_strats = ["always", "always"]

# prefix_out = "eg_7_all_d_never"
# coauthor_strats = ["all_d", "all_d"]
# fsauthor_strats = ["never", "never"]

# prefix_out = "eg_7_pavlov_d_always"
# coauthor_strats = ["pavlov_d", "pavlov_d"]
# fsauthor_strats = ["always", "always"]

# prefix_out = "eg_7_pavlov_c_always"
# coauthor_strats = ["pavlov_c", "pavlov_c"]
# fsauthor_strats = ["always", "always"]

# prefix_out = "eg_7_pavlov_d_never"
# coauthor_strats = ["pavlov_d", "pavlov_d"]
# fsauthor_strats = ["never", "never"]

# prefix_out = "eg_7_pavlov_c_never"
# coauthor_strats = ["pavlov_c", "pavlov_c"]
# fsauthor_strats = ["never", "never"]

prefix_out = "eg_7_all_c_never"
coauthor_strats = ["all_c", "all_c"]
fsauthor_strats = ["never", "never"]

# prefix_out = "eg_7_all_c_always"
# coauthor_strats = ["all_c", "all_c"]
# fsauthor_strats = ["always", "always"]


# parameters
# ---

n = 2  # number of players
b = 2  # maximum benefit of a paper
c_f = 1  # cost of first-authoring
# c_c = 0.8  # cost of co-authoring
c_c = 1.8  # cost of co-authoring

# topic preferences
preferences = np.array([[0], [1]])
assert len(preferences) == n


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

ID_2_actions = mf.get_ID_2_actions(params)
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
frsauthor_2_fillcolor = {
    (0, 0): "#ffcccc",
    (0, 1): "#eeeebb",
    (1, 0): "#ccddaa",
    (1, 1): "#bbccee",
}
ID_2_fillcolor = {
    ID: frsauthor_2_fillcolor[tuple([actions[i][i] for i in range(n)])]
    for ID, actions in ID_2_actions.items()
}


AG = mf.create_attributes_deterministic_graph(G, ID_2_actions, ID_2_fillcolor)
mf.print_attributes_deterministic_graph(AG, prefix_out)
