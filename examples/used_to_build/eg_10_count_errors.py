# given a deterministic transition matrix,
# find the stationary distribution as eps -> 0

import examples_fncs_10 as mf
import numpy as np
import networkx as nx
import subprocess





# =============================================================================


# parameters
# ---

prefix_out = "eg_10"

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


# the scenario "co-author pavlov-c, first author never"
# has an interesting shape, so use that matrix

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


# index every possible n-player action state
# ---

ID_2_actions = mf.get_ID_2_actions(params)
nbr_poss_actions = len(ID_2_actions)
actions_2_ID = {actions: ID for ID, actions in ID_2_actions.items()}


# create matrix of valid transitions
# ---

fsco_actions_2_ID = {
    mf.actions_2_fsactions_coactions(actions): ID for actions, ID in actions_2_ID.items()
}
valid_transitions = mf.calc_valid_transitions(fsco_actions_2_ID)


# verify that every deterministic transition calculate previously is a valid transition
# ---

for pre_ID in range(nbr_poss_actions):
    for suc_ID in range(nbr_poss_actions):
        if deterministic_transitions[pre_ID][suc_ID] == 1:
            if valid_transitions[pre_ID][suc_ID] == 0:
                print(f"Invalid deterministic transitions: ({pre_ID}, {suc_ID})")
                break
# -- all good


# count the number of errors to get from each predecessor to ultimate successor
# ---

ID_2_fsco_actions = {ID: fsco_actions for fsco_actions, ID in fsco_actions_2_ID.items()}

nbr_errorsM = [[None] * nbr_poss_actions for _ in range(nbr_poss_actions)]
nbr_correctM = [[None] * nbr_poss_actions for _ in range(nbr_poss_actions)]

for pre_ID in range(nbr_poss_actions):

    # the comparison is made to the deterministic successor
    det_ID = deterministic_transitions[pre_ID].index(1)
    det_fsactionV, det_coactionsV = ID_2_fsco_actions[det_ID]

    for ult_ID in range(nbr_poss_actions):

        # only count errors between valid transitions
        if valid_transitions[pre_ID][ult_ID] == 1:

            # the ultimate predecessor in a convenient form
            ult_fsactionV, ult_coactionsV = ID_2_fsco_actions[ult_ID]

            # count all first-author differences
            nbr_fs_errors = sum(
                ult_fsaction != det_fsaction
                for ult_fsaction, det_fsaction in zip(ult_fsactionV, det_fsactionV)
            )

            # count only those co-author differences that occur where the 
            # first author is authoring in the ultimate successor

            # indices where first author is authoring
            fs1_idxs = [idx for idx, fsaction in enumerate(ult_fsactionV) if fsaction == 1]

            # count only those rows (co-authorship relationships)
            nbr_co_errors = sum(
                sum(
                    ult_coaction != det_coaction
                    for ult_coaction, det_coaction in zip(ult_coactionsV[fs1_idx], det_coactionsV[fs1_idx])
                )
                for fs1_idx in fs1_idxs
            )

            # get the number of errors, number correct, and store
            nbr_errors = nbr_fs_errors + nbr_co_errors
            nbr_errorsM[pre_ID][ult_ID] = nbr_errors
            nbr_correctM[pre_ID][ult_ID] = n + len(fs1_idxs) * (n - 1) - nbr_errors


# plot one transition, predecessor to all its successors, deterministic and erroneous
# ---

pre_IDs = [0, 1, 2, 4, 7, 9, 10, 14, 15]


for pre_ID in pre_IDs:

    # create an attributes dictionary with the number of errors for each transition
    attrs = {
        (pre_ID, ult_ID): {
            "label": f"{nbr_errorsM[pre_ID][ult_ID]}, {nbr_correctM[pre_ID][ult_ID]}",
        }
        for ult_ID, valid in enumerate(valid_transitions[pre_ID])
        if valid
    }

    # highlight deterministic transition in blue
    det_ID = deterministic_transitions[pre_ID].index(1)
    attrs[(pre_ID, det_ID)]["color"] = "blue"

    # create the networkx graph
    G = nx.DiGraph()
    for pre_ID, ult_ID in attrs:
        G.add_edge(pre_ID, ult_ID)

    nx.set_edge_attributes(G, attrs) # add edge attributes

    # turn into node-attributed graph and print
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
    mf.print_attributes_deterministic_graph(AG, f"{prefix_out}_from_{pre_ID}")

