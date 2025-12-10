# Write the full in-game deterministic transition matrix,
#
# same as eg_4 but doing pavlov
# updated code to account for pavlov when one wasn't first-authoring

import numpy as np
import itertools as it
import networkx as nx

import examples_fncs_2 as mf


# --------------------------------

# parameters
# ---

# NOTE lazy commenting out scenarios here

#prefix_out = "eg_6_pavlov_c"
#coauthor_strats = ["pavlov-c", "pavlov-c"]

prefix_out = "eg_6_pavlov_d"
coauthor_strats = ["pavlov-d", "pavlov-d"]



# coauthor_strats = ["all_d", "all_d"]
fsauthor_strats = ["always", "always"]

n = 2  # number of players
b = 2  # maximum benefit of a paper
c_f = 1  # cost of first-authoring
# c_c = 0.8  # cost of co-authoring
c_c = 1.8  # cost of co-authoring

# topic preferences
preferences = np.array(
    [
        np.array([0]),
        np.array([1]),
    ],
    dtype=float,
)
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
    "possible_coauthor_strats": ["all_c", "all_d", "pavlov-c", "pavlov-d"],
    "possible_fsauthor_strats": ["always"],
}


# list every possible n-player action state
# ---

n = params["nbr_players"]

# the possible action-states for player i
# can be encoded as every possible binary string
# of length n, where element j == a_{i,j}
#
# e.g., if n = 2, possible action-states are:
# [(0, 0), (0, 1), (1, 0), (1, 1)]
#
# len(poss_action) = 2**n

poss_action = list(it.product([0, 1], repeat=n))

# the possible action-states for all players
# is the product of possible action-states
# for each of the n players

# len(poss_actions) = [2**n]**n = 2**(n**2)
poss_actions = list(it.product(poss_action, repeat=n))
nbr_poss_actions = len(poss_actions)
assert nbr_poss_actions == 2 ** (n**2)

# make handy dictionaries so we can refer to each actions-state by its ID
actions_2_ID = {actions: ID for ID, actions in enumerate(poss_actions)}
ID_2_actions = dict(enumerate(poss_actions))


# parameters with fixed values (depends on model type)
# ---

# the simplifying assumptions of this model mean gammas and alignments are constant
gammas = [[mf.gamma_fnc(params, i, j) for j in range(n)] for i in range(n)]
topics = mf.topics_fnc(params)
alignments = mf.alignments_fnc(params, preferences, topics)


# for each possible n-player actions-state, the action switching gains
# ---

# gains that are positive mean that, for authorship in that instance (first- or co-authorship),
# if the focal player unilaterally switches from not-author to author (a_{i,j} switches from 0 to 1),
# then the direct authorship payoffs exceed the costs
#
# in other words, a player should always author in this situation

action_gains_positive = {
    ID: [
        [action_gain > 0 for action_gain in focal_action_gains]
        for focal_action_gains in mf.calc_action_gains(
            params, prev_actions, gammas, alignments
        )
    ]
    for ID, prev_actions in ID_2_actions.items()
}


# for each possible n-player action-state, calculate the deterministic next state
# ---

# takes into account both the unilateral decisions above and the focal's
# authorship strategies

deterministic_transitions = [[0] * nbr_poss_actions for _ in range(nbr_poss_actions)]
for prev_ID, prev_actions in ID_2_actions.items():
    # for this previous-actions matrix, get the next-actions matrix
    next_actions = [[None] * n for _ in range(n)]

    for other in range(n):
        # updating rules depend on whether the other player is publishing
        other_published = prev_actions[other][other] == 1

        # get the next act of focal to other
        for focal in range(n):
            if focal == other:
                # first authors are unconstrained in their actions

                if action_gains_positive[prev_ID][focal][other]:
                    # if first-authorship has a positive unilateral switching gain,
                    # then first-authorship will be pursued;
                    next_act = 1

                else:
                    # if first-authorship doesn't have a positive switching gain,
                    # the decision depends on the focal's first-authorship strategy

                    # the previous action is whether the focal first-authored (1)
                    # or not (0)
                    prev_act = prev_actions[focal][other]

                    # the strategy is a string that is used by the next-act
                    # calculation
                    fsauthor_strat = fsauthor_strats[focal]
                    next_act = mf.calc_next_fsauthor_act(prev_act, fsauthor_strat)

            else:
                # co-authorship rules depend on whether the other is publishing or not

                if not other_published:
                    # if the other is not publishing, then focal's coauthorship
                    # strategy cannot not change -- represents perfect "memory"
                    next_act = prev_actions[focal][other]

                else:
                    # if the other is publishing, then focal's coauthorship unconstrained

                    if action_gains_positive[prev_ID][focal][other]:
                        # if co-authorship has a positive unilateral switching gain,
                        # then co-authorship will be pursued;
                        next_act = 1

                    else:
                        # if co-authorship does not have a positive
                        # switching gain, then co-authorship action
                        # depends on: focal's and others' prev action
                        # and focal's co-author strategy
                        prev_ff, prev_fo, prev_of, prev_oo = (
                            prev_actions[focal][focal],
                            prev_actions[focal][other],
                            prev_actions[other][focal],
                            prev_actions[other][other],
                        )

                        # co-author strategy string
                        coauthor_strat = coauthor_strats[focal]
                        next_act = mf.calc_next_coauthor_act(
                            prev_ff, prev_fo, prev_of, prev_oo, coauthor_strat
                        )

            # store next-act of focal to other in the next-actions matrix
            next_actions[focal][other] = next_act

    # next-actions matrix is now populated

    # identify which actions index this next-actions matrix corresponds to
    next_actions = tuple(tuple(inner_list) for inner_list in next_actions)
    next_ID = actions_2_ID[next_actions]

    # store the transition from previous matrix ID to next matrix ID
    deterministic_transitions[prev_ID][next_ID] = 1


# plot to check
# ---

# create the networkx graph
# ---

G = nx.DiGraph(np.array(deterministic_transitions))


# define fill colours for each action-state
# ---

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
