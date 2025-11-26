# given a deterministic transition matrix,
# find the stationary distribution as eps -> 0

import examples_fncs_3 as mf
import numpy as np
import networkx as nx


def actions_2_fsactions_coactions(actions):
    # Convert an actions matrix (list or tuple) to two tuples:
    # (1) first-author actions, and (2) co-author actions.
    #
    #
    # Inputs:
    # ---
    #
    # actions, n x n binary list of lists
    #   Each element a_{i,j} is the action of potential author i with respect
    #   to the paper first-authored by j
    #
    #
    # Outputs:
    # ---
    #
    # fs_actionV, binary tuple of length n
    #   First-author actions. Each element i indicates whether player i
    #   first-authored (1) or not (0).
    #
    # co_actionsV, (n-1) x (n-1) binary tuple of tuples
    #   Co-author actions. Each row i corresponds to the first author, and each
    #   column corresponds to the co-author, however the col indexing skips the
    #   co-author = first-author case.
    #
    #
    # Example:
    # ---
    #
    # >>> matrix = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
    # >>> actions_2_fsactions_coactions(matrix)
    # (('a', 'e', 'i'), (('d', 'g'), ('b', 'h'), ('c', 'f')))

    n = len(actions)
    assert len(actions[0]) == n

    # diagonal elements are first-author actions
    fs_actionV = tuple([actions[i][i] for i in range(n)])

    # off-diagonal elements are co-author actions
    # transpose them so each row corresponds to a player in fs_actionV
    co_actionsV = tuple(
        [tuple([actions[co][fs] for co in range(n) if co != fs]) for fs in range(n)]
    )

    return fs_actionV, co_actionsV


def fsactions_coactions_2_actions(fs_actionV, co_actionsV):
    # Convert two tuples --- (1) first-author actions, and (2) co-author
    # actions --- into an actions matrix (tuple of tuples)
    #
    #
    # Inputs:
    # ---
    #
    # fs_actionV, binary tuple of length n
    #   First-author actions. Each element i indicates whether player i
    #   first-authored (1) or not (0).
    #
    # co_actionsV, (n-1) x (n-1) binary tuple of tuples
    #   Co-author actions. Each row i corresponds to the first author, and each
    #   column corresponds to the co-author, however the col indexing skips the
    #   co-author = first-author case.
    #
    #
    # Outputs:
    # ---
    #
    # actions, n x n binary tuple of tuples
    #   Each element a_{i,j} is the action of potential author i with respect
    #   to the paper first-authored by j
    #
    #
    # Example:
    # ---
    #
    # >>> fs_actionV = ('a', 'e', 'i')
    # >>> co_actionsV = (('d', 'g'), ('b', 'h'), ('c', 'f'))
    # >>> fsactions_coactions_2_actions(fs_actionV, co_actionsV)
    # (('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'h', 'i'))

    n = len(fs_actionV)
    assert len(co_actionsV) == n
    assert len(co_actionsV[0]) == (n - 1)

    # remember the co_actionsV are columns of the actions matrix
    # and the focal index is skipped

    # actions matrix transposed
    actions_transposed = [
        co_actions[:fs_idx] + (fs_action,) + co_actions[fs_idx:]
        for fs_idx, (fs_action, co_actions) in enumerate(zip(fs_actionV, co_actionsV))
    ]

    # transpose to get original
    actions = tuple(zip(*actions_transposed))

    return actions


# =============================================================================


# parameters
# ---

prefix_out = "eg_8"

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

# also make a dictionary of the (fsaction, coactions) form,
# which is more convenient for the calculation below
# (code from eg_9.py)

fsco_actions_2_ID = {
    actions_2_fsactions_coactions(actions): ID for actions, ID in actions_2_ID.items()
}


# create matrix of valid transitions
# ---

# NOTE: this matrix is universal to all 2-player games
valid_transitions = [[None] * nbr_poss_actions for _ in range(nbr_poss_actions)]

for (suc_fsactionV, suc_coactionsV), suc_ID in fsco_actions_2_ID.items():
    # for each successor

    # identify which players are not first-authoring in the successor
    fs0_idxs = [idx for idx, fsaction in enumerate(suc_fsactionV) if fsaction == 0]

    # identify the co-authorship memories associated with non-first-authoring player
    suc_coactionsV_fs0 = [suc_coactionsV[fs0_idx] for fs0_idx in fs0_idxs]

    for (pre_fsactionV, pre_coactionsV), pre_ID in fsco_actions_2_ID.items():
        # for each predecessor

        # transition from the predecessor to the successor is only permissible
        # if the memories in the successor match the memories/actions in the
        # predecessor

        # identify which players are not first-authoring in the predecessor
        pre_coactionsV_fs0 = [pre_coactionsV[fs0_idx] for fs0_idx in fs0_idxs]

        if pre_coactionsV_fs0 != suc_coactionsV_fs0:
            valid_transitions[pre_ID][suc_ID] = 0
        else:
            valid_transitions[pre_ID][suc_ID] = 1


# verify that every deterministic transition calculate previously is a valid transition
# ---

for pre_ID in range(nbr_poss_actions):
    for suc_ID in range(nbr_poss_actions):
        if deterministic_transitions[pre_ID][suc_ID] == 1:
            if valid_transitions[pre_ID][suc_ID] == 0:
                print(f"Invalid deterministic transitions: ({pre_ID}, {suc_ID})")
                break
# -- all good


# plot to check
# ---

# create the networkx graph
G = nx.DiGraph(np.array(valid_transitions))

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
