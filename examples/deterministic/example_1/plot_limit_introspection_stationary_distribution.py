# Given the expected payoffs between strategies, plot a graph of 
# the introspection-dynamics in the limit when introspection 
# strength delta -> oo
#
# Note that this only looks at dynamics of co-author strategy,
# and first-author strategy is assumed fixed ("never").
#
# For future work, maybe don't write predecessor results 
# taking advantage of the symmetry between players.

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
import sympy as sp
import itertools as it
import networkx as nx
import tol_colors as tc

import model_fncs as mf


# parameters
# ---

# where the expected payoffs are stored
path_in = "within_game_stationary_payoffs.parquet"

# where to store the dot file
path_out = "limit_introspection_stationary_distribution.dot"


# local functions 
# ---

# probability of transition as a function of old versus new payoffs 
# given that the focal player and the comparison player have already 
# beeen chosen
# def prob_transition_given_comparison_pair(focal_idx, old_pays, new_pays):
    # return 1 / (1 + np.exp(-delta * (new_pays[focal_idx] - old_pays[focal_idx])))


# create useful dictionaries 
# ---

# read in the expected payoff to each strategy in each strategy-pair scenario 
table = pq.read_table(
        path_in,
        columns = [
            "player_1_coauthor_strategy", 
            "player_2_coauthor_strategy", 
            "player_1_payoff", 
            "player_2_payoff", 
        ]
)

# use table to make dictionary from strategy pair to payoff pair 
strats_2_pays = {
        (row["player_1_coauthor_strategy"], row["player_2_coauthor_strategy"]):
        (row["player_1_payoff"], row["player_2_payoff"])
        for row in table.to_pylist()
}
# note this dictionary takes advantage of the symmetry between players,
# listing only combinations-with-replacement pairs of strategies 
# rather than the full product of strategy pairs

# dictionaries indexing the coauthor strategies 
ordered_strats = sorted(set(
    table["player_1_coauthor_strategy"].to_pylist() 
    + table["player_1_coauthor_strategy"].to_pylist()
))
nbr_strats = len(ordered_strats)
strat_idx_2_strat = dict(enumerate(ordered_strats))
strat_2_strat_idx = {strat: idx for idx, strat in strat_idx_2_strat.items()}

# indexation of the transition matrix 
mat_idx_2_strat_idxs = dict(enumerate(list(it.product(strat_idx_2_strat.keys(), repeat=2))))
strat_idxs_2_mat_idx = {strat_idxs: mat_idx for mat_idx, strat_idxs in mat_idx_2_strat_idxs.items()}

# a dictionary mapping from coauthor-strategy index pair to payoffs pair
# this dictionary contains the full product
mat_idx_2_pays = dict()
for mat_idx, strat_idxs in mat_idx_2_strat_idxs.items():
    strats = (strat_idx_2_strat[strat_idxs[0]], strat_idx_2_strat[strat_idxs[1]])
    if strats in strats_2_pays:
        mat_idx_2_pays[mat_idx] = strats_2_pays[strats]
    else:
        # the strategy payoffs were stored taking advantage of symmetry between players
        pays_rev = strats_2_pays[(strats[1], strats[0])]
        mat_idx_2_pays[mat_idx] = (pays_rev[1], pays_rev[0])


# create the graph
# ---

G = nx.DiGraph()
mat_len = len(mat_idx_2_strat_idxs)

# populate the transitions between states
for old_mat_idx in range(mat_len):

    old_strat_idxs = mat_idx_2_strat_idxs[old_mat_idx]
    old_pays = mat_idx_2_pays[old_mat_idx]

    # players only update their strategy one at a time, so the only 
    # permissible transitions from (strat_player_1, strat_player_2) = (k, l)
    # are (k, l) -> (k', l) and (k, l) -> (k, l')

    # permissible transitions for player 1
    p1_old_strat_idx = old_strat_idxs[0]
    p1_permissible_new_strat_idxs = [
        (new_strat_idx, old_strat_idxs[1]) for new_strat_idx in range(nbr_strats) 
        if new_strat_idx != p1_old_strat_idx
    ]

    # permissible transitions for player 2
    p2_old_strat_idx = old_strat_idxs[1]
    p2_permissible_new_strat_idxs = [
        (old_strat_idxs[0], new_strat_idx) for new_strat_idx in range(nbr_strats) 
        if new_strat_idx != p2_old_strat_idx
    ]

    permissible_new_strat_idxs = [p1_permissible_new_strat_idxs, p2_permissible_new_strat_idxs]

    for focal_idx in [0, 1]:
        for new_strat_idxs in permissible_new_strat_idxs[focal_idx]:
            new_mat_idx = strat_idxs_2_mat_idx[new_strat_idxs]
            new_pays = mat_idx_2_pays[new_mat_idx]

            if new_pays[focal_idx] >= old_pays[focal_idx]:
                G.add_edge(old_mat_idx, new_mat_idx)


# plot a graph of the full transitions
# ---

# label the nodes
node_attrs = {
        mat_idx: {
            "label":
            f"{mat_idx}: {strat_idxs}\n" + ", ".join([strat_idx_2_strat[strat_idxs[0]], strat_idx_2_strat[strat_idxs[1]]]),
        }
        for mat_idx, strat_idxs in mat_idx_2_strat_idxs.items()
}

# make each strongly connected component a different colour
sccs = list(nx.strongly_connected_components(G))
cset = list(tc.pale)
nbr_cset = len(cset)

for c_idx, component in enumerate(sccs):
    colour = cset[c_idx % nbr_cset] # just loop if more than
    for mat_idx in component:
        node_attrs[mat_idx]["fillcolor"] = colour

# set node attributes
nx.set_node_attributes(G, node_attrs)

# print attribute graph 
AG = nx.nx_agraph.to_agraph(G)
AG.graph_attr["overlap"] = "false" # global attributes
AG.graph_attr["d2tgraphstyle"] = "every node/.style={draw}"
AG.node_attr["style"] = "filled"
AG.node_attr["shape"] = "circle"
AG.node_attr["color"] = "white"
AG.write(path_out) # write to dot file


# plot condensation graph
# ---

C = nx.condensation(G)

# label each node with its list of nodes in the original graph,
# and match the colours
node_attrs = {c_idx: dict() for c_idx in C.nodes()}
for c_idx, component in enumerate(sccs):

    # colour with matching colour
    colour = cset[c_idx % nbr_cset] # just loop if more than
    node_attrs[c_idx]["fillcolor"] = colour

    # label with list of original nodes 
    node_attrs[c_idx]["label"] = ", ".join([str(v) for v in sorted(component)])

nx.set_node_attributes(C, node_attrs)

# print attribute graph of condensation graph
AG = nx.nx_agraph.to_agraph(C)
AG.graph_attr["overlap"] = "false" # global attributes
AG.graph_attr["d2tgraphstyle"] = "every node/.style={draw}"
AG.node_attr["style"] = "filled"
AG.node_attr["shape"] = "circle"
AG.node_attr["color"] = "white"
old_path_out = path_out.split(".")
new_path_out = old_path_out[0] + "_condensation.dot"
AG.write(new_path_out) # write to dot file

