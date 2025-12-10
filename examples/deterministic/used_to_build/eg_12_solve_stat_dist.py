# let's see if we're ready to solve the stationary distribution
#
# the deterministic state I use is the solution to
# co-authors pavlov-c, first-authors never

import pandas as pd
import sympy as sp
import numpy as np
import networkx as nx

import examples_fncs_11 as mf


# parameters
# ---

prefix_out = "eg_12"

n = 2

deterministic_transitions = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]


# list the deterministic successors
# ---

pre_2_det = {
    pre: det_row.index(1)
    for pre, det_row in enumerate(deterministic_transitions)
}


# read in the 2-player transitions and nbr errors
# ---

df = pd.read_csv("transitions_errors_nbr_players_2.csv")
pre_2_suc = {pre: dict() for pre in df["from_state"]}
for pre, suc, is_valid, nbr_errors, nbr_correct in zip(df["from_state"], df["to_state"], df["is_valid"], df["nbr_errors"], df["nbr_correct"]):
    pre_2_suc[pre][suc] = {
        "is_valid": is_valid,
        "nbr_errors_correct": (nbr_errors, nbr_correct)
    }


# find the basins of attraction
# ---

# create the networkx graph
det_G = nx.DiGraph(np.array(deterministic_transitions))

nbr_states = len(deterministic_transitions)

# absorbing states NOTE: being lazy
absorbs = [
    i for i in range(nbr_states)
    if deterministic_transitions[i][i] == 1
]

idx_2_attractor = {
    idx: attractor
    for idx, attractor in enumerate(nx.algorithms.components.attracting_components(det_G))
}
nbr_attractors = len(idx_2_attractor)

# the full basin is the attractor + its basin of attraction
attractor_idx_2_full_basin = {
    idx: attractor.union(nx.ancestors(det_G, next(iter(attractor))))
    for idx, attractor in idx_2_attractor.items()
}

# useful reverse dictionary
state_2_attractor_idx = {
    state: attractor_idx
    for attractor_idx, states in attractor_idx_2_full_basin.items()
    for state in states
}

# NOTE lazily assuming one absorbing state per attractor
absorb_2_idx = {
    absorb: idx
    for idx, absorbs in idx_2_attractor.items()
    for absorb in absorbs
}
idx_2_absorb = {
    idx: absorb for absorb, idx in absorb_2_idx.items()
}


# make a graph between the absorbing states only for error-1s only
# ---

# count number of error edges between them
nbr_absorbs = len(idx_2_absorb)
eps1s = [[0]*nbr_absorbs for _ in range(nbr_absorbs)]

for i, absorb in enumerate(absorbs):
    
    # which attractor index is this?
    absorb_idx = absorb_2_idx[absorb]

    for suc, sucD in pre_2_suc[absorb].items():
        
        # if it takes only 1 error to get there
        if sucD["is_valid"]:
            if sucD["nbr_errors_correct"][0] == 1:

                attract_suc_idx = state_2_attractor_idx[suc]

                # and if it leads to a different attractor
                if attract_suc_idx != absorb_idx:
                    eps1s[absorb_idx][attract_suc_idx] += 1

# create the networkx graph
G = nx.DiGraph(np.array(eps1s))

# define fill colours for each action-state
frsauthor_2_fillcolor = {
    (0, 0): "#ffcccc",
    (0, 1): "#eeeebb",
    (1, 0): "#ccddaa",
    (1, 1): "#bbccee",
}

ID_2_actions = mf.get_ID_2_actions(n)
absorb_ID_2_actions = {
    absorb_2_idx[ID]: ID_2_actions[ID] for ID in absorbs
}
absorb_ID_2_fillcolor = {
    ID: frsauthor_2_fillcolor[tuple([actions[i][i] for i in range(n)])]
    for ID, actions in absorb_ID_2_actions.items()
}

# add edge labels
for (u, v) in G.edges():
    nbr_errors = eps1s[u][v]
    str_nbr_errors = "" if nbr_errors == 1 else str(nbr_errors)
    G[u][v].update(
        color="red",
        lblstyle="above=0.3cm, red, draw = none",
        texlbl = str_nbr_errors + r"$\varepsilon$", 
    )


AG = mf.create_attributes_deterministic_graph(G, absorb_ID_2_actions, absorb_ID_2_fillcolor)
mf.print_attributes_deterministic_graph(AG, f"{prefix_out}_with_errors_simpler", "dot")


# plot the transition graph including errors
# ---

# create the networkx graph for the deterministic transitions


# define fill colours for each action-state
frsauthor_2_fillcolor = {
    (0, 0): "#ffcccc",
    (0, 1): "#eeeebb",
    (1, 0): "#ccddaa",
    (1, 1): "#bbccee",
}

ID_2_actions = mf.get_ID_2_actions(n)
ID_2_fillcolor = {
    ID: frsauthor_2_fillcolor[tuple([actions[i][i] for i in range(n)])]
    for ID, actions in ID_2_actions.items()
}

# add red edges for transitions from every absorbing state
# to an error of order epsilon 1

# the absorbing states
# NOTE: I'm lazily assuming they're self-loops here,
# but that might not always be true


# show all the deterministic edges and all error-1s
# ---

# for each absorbing state,
# add a red edge from it to all its nbr_errors = 1 states
# but only if it ends up in a different basin

G_all = det_G.copy()

for absorbing in absorbs:
    
    # which attractor index is this?
    attractor_idx = state_2_attractor_idx[absorbing]

    for suc, sucD in pre_2_suc[absorbing].items():
        
        # if it takes only 1 error to get there
        if sucD["is_valid"]:
            if sucD["nbr_errors_correct"][0] == 1:

                # and if it leads to a different attractor
                if state_2_attractor_idx[suc] != attractor_idx:
                    G_all.add_edge(
                        absorbing, 
                        suc, 
                        color="red",
                        lblstyle="above=0.3cm, red, draw = none",
                        texlbl = r"$\varepsilon$", 
                )

# create the attributes graph and print
AG = mf.create_attributes_deterministic_graph(G_all, ID_2_actions, ID_2_fillcolor)
mf.print_attributes_deterministic_graph(AG, f"{prefix_out}_with_errors", "dot")


# find the stationary distribution as eps -> 0
# ---

# list the deterministic successors
pre_2_det = {
    pre: det_row.index(1) for pre, det_row in enumerate(deterministic_transitions)
}

# get the transition matrix with error terms included (symbolic entries a fnc of eps)
P, eps = mf.symbolic_transitions_with_errors(pre_2_det, "transitions_errors_nbr_players_2.csv")

# find the stationary distribution as eps -> 0
# max_pwr = 2 # maximimum power level of terms to include, defaulted
stationary_distn, pwr_level = mf.find_stationary_distribution(P, eps)
