# put the functions used in the examples for here for now
#
# doing the error transitions now
#
# updates:
# - make it so get_ID_2_actions() accepts n not params
# - add symbolic P and stationary distribution fncs

from scipy.spatial import distance

import networkx as nx
import subprocess
import sympy as sp
import pandas as pd
import numpy as np

import itertools as it


def pg_indicator_fnc(params, a_ij):
    type_of_good = params["type_of_good"]
    if type_of_good == "excludable":
        return a_ij
    elif type_of_good == "non-excludable":
        return 1
    else:
        raise ValueError(
            f"Invalid type_of_good: '{type_of_good}'. Expected 'excludable' or 'non-excludable'."
        )


def gamma_fnc(params, i, j):
    gamma_rule = params["gamma_rule"]
    if gamma_rule == "constant_1":
        return 1
    else:
        raise ValueError(
            f"Invalid gamma_rule: '{gamma_rule}'. Expected 'constant_1' or ... (to write)."
        )


def papers_influences_fnc(params, actions):
    # Inputs:
    # ---
    # actions, n x n list of lists of 1s and zeros:
    #   actions[i][j] is 1 if player i contributes to player j's paper

    # extract parameters
    wf = params["first_author_weight"]
    b = params["max_paper_benefit"]
    n = params["nbr_players"]

    # calculate paper's influence

    # nbr of coauthors of each paper (not counting first author)
    nbr_coauthors = [sum(actions[j][i] for j in range(n) if i != j) for i in range(n)]

    # paper influence calculation
    influences = [
        b * (wf + nbr_coauthor) / (wf + n - 1) for nbr_coauthor in nbr_coauthors
    ]

    return influences


def paper_influence_variation(params, actions):
    # Calculate the paper influences that would result
    # in the situation where each focal co-author
    # does not co-author the paper or does

    # extract parameters
    wf = params["first_author_weight"]
    b = params["max_paper_benefit"]
    n = params["nbr_players"]

    # nbr of co-authors for each paper apart from the
    # focal coauthor k (rows) and first author (col)
    nbr_nonfocal_coauthors = [
        [
            sum(actions[j][i] for j in range(n) if (j != i) and (j != k))
            for i in range(n)  # paper first authors
        ]
        for k in range(n)  # focal co-authors
    ]

    # NOTE: this would be quicker with np arrays

    # calculate the influences that would result
    # if the focal coauthor k (row) did not coauthor
    influences_0 = [
        [
            b * (wf + nbr_nonfocal_coauthors[k][i]) / (wf + n - 1)
            if i != k
            else None  # doesn't make sense to calc for first author
            for i in range(n)  # paper first authors
        ]
        for k in range(n)  # focal coauthors
    ]

    # calculate the influences that would result
    # if the focal coauthor k (row) did coauthor
    influences_1 = [
        [
            b * (wf + nbr_nonfocal_coauthors[k][i] + 1) / (wf + n - 1)
            if i != k
            else None  # doesn't make sense to calc for first author
            for i in range(n)  # paper first authors
        ]
        for k in range(n)  # focal coauthors
    ]

    return (influences_0, influences_1)


def alignments_fnc(params, preferences, topics):
    # Calculate the alignment between coauthor preferences
    # and first-author paper topics
    #
    # Inputs:
    # ---
    # to od
    #
    # Outputs:
    # ---
    #
    # alignments, list of lists
    #   Row indices correspond to coauthors, column indices to
    #   paper topics, and values are the alignment

    # extract key parameters
    distance_measure = params["distance_measure"]
    n = params["nbr_players"]

    # different calculations are made depending on the measure
    # of distance between positions (coauthor preference and
    # paper topic)
    #
    # alignment = 1 when distance = 0,
    # alignment -> 0 as distance -> oo

    if distance_measure == "euclidean":
        alignments = [
            [
                float(
                    1
                    / (
                        1
                        + distance.euclidean(
                            preferences[co_author], topics[first_author]
                        )
                    )
                )
                for first_author in range(n)
            ]
            for co_author in range(n)
        ]

    else:
        raise ValueError(
            f"Invalid distance_measure: '{distance_measure}'. Expected 'euclidean' or ..."
        )

    return alignments


def authoring_cost_fnc(params, i, j):
    authoring_cost_rule = params["authoring_cost_rule"]

    if authoring_cost_rule == "simple":
        if i == j:
            authoring_cost = params["first_author_cost"]
        else:
            authoring_cost = params["coauthor_cost"]
    else:
        raise ValueError(
            f"Invalid authoring_cost_rule: '{authoring_cost_rule}'. Expected 'simple' or ..."
        )

    return authoring_cost


def topics_fnc(params):
    topic_choice_rule = params["topic_choice_rule"]
    preferences = params["preferences"]

    if topic_choice_rule == "exact_preference":
        topics = preferences
    else:
        raise ValueError(
            f"Invalid topic_choice_rule: '{topic_choice_rule}'. Expected 'simple' or ..."
        )

    return topics


def calc_pays(params, actions, topics=None):
    # Calculate the payoff to each player given
    # their actions and the topics

    # extract needed parameters
    n = params["nbr_players"]
    preferences = params["preferences"]

    if topics is None:
        # topics are fixed and defined in params
        topics = topics_fnc(params)

    # rows: co-authors' preferences
    # colums: first-authors' topics
    alignments = alignments_fnc(params, preferences, topics)

    # coefficient accounting for effect of authorship role on benefit
    gammas = [[gamma_fnc(params, i, j) for j in range(n)] for i in range(n)]

    # papers' influence levels
    influences = papers_influences_fnc(params, actions)

    # payoff calculation for each player
    pays = [
        sum(
            actions[j][j]
            * (
                pg_indicator_fnc(params, actions[i][j])
                * gammas[i][j]
                * influences[j]
                * alignments[i][j]
                - actions[i][j] * authoring_cost_fnc(params, i, j)
            )
            for j in range(n)
        )
        for i in range(n)  # for each first-author
    ]

    return pays


def print_payoff_matrix_elements(params):
    print("Payoff-matrix elements")

    # list out each of the actions possible in the simple model
    # keys refer to the location in the payoff matrix
    possible_actions = {
        "a": [[1, 1], [1, 1]],
        "b": [[1, 1], [0, 1]],
        "c": [[1, 0], [1, 1]],
        "d": [[1, 0], [0, 1]],
    }

    pm = {
        element: calc_pays(params, actions)[0]
        for element, actions in possible_actions.items()
    }
    print(pm)

    # determine which regime we're in
    if pm["a"] > pm["b"] > pm["c"] > pm["d"]:
        print("Mutualism 1")
    elif pm["a"] > pm["c"] > pm["b"] > pm["d"]:
        print("Mutualism 2")
    elif pm["c"] > pm["a"] > pm["d"] > pm["b"]:
        print("Prisoner's Dilemma")
    elif pm["c"] > pm["d"] > pm["a"] > pm["b"]:
        print("Never cooperator")
    else:
        print("Unknown regime??")


def calc_next_pavlov_act(prev_fo, prev_of):
    # Calculate co-authorship next action based on the Pavlov strategy
    if ((prev_fo, prev_of) == (0, 0)) or ((prev_fo, prev_of) == (1, 1)):
        next_act = 1

    elif ((prev_fo, prev_of) == (1, 0)) or ((prev_fo, prev_of) == (0, 1)):
        next_act = 0

    return next_act


def calc_next_coauthor_act(prev_ff, prev_fo, prev_of, prev_oo, strategy_name):
    # Calculate each co-authorship next action based on
    # previous actions and the coauthorship strategy name
    #
    # Inputs:
    # ---
    #
    # prev_ff, binary int (0 or 1)
    #   The focal's previous first-authorship action.
    #
    # prev_fo, binary int (0 or 1)
    #   The focal's previous co-authorship action wrt other.
    #
    # prev_of, binary int (0 or 1)
    #   The others's previous co-authorship action wrt focal.
    #
    # prev_oo, binary int (0 or 1)
    #   The other's previous first-authorship action.
    #
    # Outputs:
    # ---
    #
    # next_act, binary int
    #   The focal's next co-authorship action wrt to the other player.

    # error checking
    # ---

    # check we know the strategy name (error raised in loop for checking own errors)
    valid_strategy_names = ["pavlov_c", "pavlov_d", "pavlov_memory", "all_c", "all_d"]

    # calculate next act based on co-authorship strategy
    # ---

    if strategy_name == "all_c":
        next_act = 1

    elif strategy_name == "all_d":
        next_act = 0

    elif strategy_name == "pavlov_memory":
        # this rule uses the memory of the most recent co-authorships
        # to decide;
        # note:
        #   - they may be out of sync, i.e., co-authorship actions
        #     occurred at different times
        #   - they may cause a focal who never publishes to erroneously
        #     co-author with an other who never co-authors back
        #     (because of a memory of the other co-authoring before
        #     before the focal stopped first-authoring)
        next_act = calc_next_pavlov_act(prev_fo, prev_of)

    elif (strategy_name == "pavlov_c") or (strategy_name == "pavlov_d"):
        if (prev_oo == 1) and (prev_ff == 1):
            # Pavlov is a reciprocal strategy and so only applies
            # when both players first-authored a paper previously
            next_act = calc_next_pavlov_act(prev_fo, prev_of)

        else:  # (prev_oo == 0) or (prev_ff == 0):
            # Otherwise, apply the fallback rule
            if strategy_name == "pavlov_c":
                next_act = 1
            else:  # strategy_name == "pavlov-d":
                next_act = 0

    else:  # strategy_name not in valid_strategy_names:
        raise ValueError(
            f"Invalid strategy_name: '{strategy_name}'. Expected one of: {', '.join(valid_strategy_names)}."
        )

    return next_act


def calc_next_fsauthor_act(prev_act, strategy_name):
    # Calculate each first-authorship next action based on
    # previous actions and the first-author strategy name

    valid_strategy_names = ["always", "never"]

    if strategy_name == "always":
        next_act = 1
    elif strategy_name == "never":
        next_act = 0
    else:
        raise ValueError(
            f"Invalid strategy_name: '{strategy_name}'. Expected one of: {', '.join(valid_strategy_names)}."
        )

    return next_act


def get_ID_2_actions(n):
    # Make a dictionary indexing every n-player action possibility.
    #
    # I think this returns the actions in their actual binary-integer order
    # TODO: check that's true
    #
    #
    # Inputs:
    # ---
    #
    # n, int
    #   Number of players
    #
    #
    # Inputs:
    # ---
    #
    # ID_2_actions, dict {int: binary matrix n x n}
    #   A dictionary giving an ID to every possible action state


    # single-player actions

    # the possible action-states for player i
    # can be encoded as every possible binary string
    # of length n, where element j == a_{i,j}
    #
    # e.g., if n = 2, possible action-states are:
    # [(0, 0), (0, 1), (1, 0), (1, 1)]
    #
    # len(poss_action) = 2**n
    poss_action = list(it.product([0, 1], repeat=n))

    # all n-player action possibilities

    # the possible action-states for all players
    # is the product of possible action-states
    # for each of the n players

    # len(poss_actions) = [2**n]**n = 2**(n**2)
    poss_actions = list(it.product(poss_action, repeat=n))
    nbr_poss_actions = len(poss_actions)
    assert nbr_poss_actions == 2 ** (n**2)

    # make handy dictionaries so we can refer to each actions-state by its ID
    ID_2_actions = dict(enumerate(poss_actions))

    return ID_2_actions


def calc_action_gains(
    params,
    actions,
    gammas,
    alignments,
):
    # Calculate the direct payoff from unilaterally switching to author
    # to not-author for each player. A positive action gain means
    # that the focal player should author (first- or co-author) that
    # paper regardless of any strategic considerations (e.g., tit-for-tat)
    # because the payoff from authoring exceeds the cost

    # extract needed parameters
    # ---

    n = params["nbr_players"]

    # first-author switching gains
    # ---

    # calculate the direct gain in payoff from switching from don't author
    # to author (i.e., a_ij = 0 to a_ij = 1)

    influences = papers_influences_fnc(params, actions)
    fsauthor_gains = [
        gammas[i][i] * influences[i] * alignments[i][i]
        - authoring_cost_fnc(params, i, i)
        for i in range(n)
    ]

    # co-author switching gains
    # ---

    # definitely worthwhile to co-author if
    # the returns from co-authoring are positive

    # The influence of each paper when focal co-author (row)
    # varies their contribution to each paper (col).
    # Matrix influence_0 gives influences when focal does not contribute,
    # and matrix influence_1 gives influences when focal does contribute.
    influences_0, influences_1 = paper_influence_variation(params, actions)

    # the direct payoff benefit of co-authoring each paper
    coauthor_gains = [
        [
            (
                pg_indicator_fnc(params, 1) * influences_1[i][j]
                - pg_indicator_fnc(params, 0) * influences_0[i][j]
            )
            * gammas[i][j]
            * alignments[i][j]
            - authoring_cost_fnc(params, i, j)
            if i != j
            else None
            for j in range(n)
        ]
        for i in range(n)
    ]

    # merge first-author and co-author switching gains to get the action switching gains
    # ---

    # switch being modelled is from a_{i, j} = 0 to a_{i, j} = 1
    action_gains = [
        [fsauthor_gains[i] if i == j else coauthor_gains[i][j] for j in range(n)]
        for i in range(n)
    ]

    return action_gains


def calc_action_gains_positive(params, ID_2_actions, gammas=None, alignments=None):
    # Gains that are positive mean that, for authorship in that
    # instance (first- or co-authorship), if the focal player
    # unilaterally switches from not-author to author (a_{i,j}
    # switches from 0 to 1), then the direct authorship payoffs
    # exceed the costs
    #
    # example for a 2-player system
    # {
    #    0: [[False, False], [False, False]],
    #    1: [[False, False], [False, False]],
    #    2: [[True, False], [False, False]],
    #    3: [[True, False], [False, False]],
    #    4: [[False, False], [False, True]],
    #       ...
    #   14: [[True, False], [False, True]],
    #   15: [[True, False], [False, True]]
    # }

    # calculate optional components if not given
    # ---

    n = params["nbr_players"]
    preferences = params["preferences"]

    if gammas is None:
        gammas = [[gamma_fnc(params, i, j) for j in range(n)] for i in range(n)]

    if alignments is None:
        topics = topics_fnc(params)
        alignments = alignments_fnc(params, preferences, topics)

    # create the matrix of which action gains are positive
    # ---

    action_gains_positive = {
        ID: [
            [action_gain > 0 for action_gain in focal_action_gains]
            for focal_action_gains in calc_action_gains(
                params, prev_actions, gammas, alignments
            )
        ]
        for ID, prev_actions in ID_2_actions.items()
    }

    return action_gains_positive


def calc_deterministic_transitions(
    params,
    actions_2_ID,
    action_gains_positive,
    fsauthor_strats,
    coauthor_strats,
):
    # extract needed parameters
    # ---

    n = params["nbr_players"]
    nbr_poss_actions = len(actions_2_ID)

    # calculate the deterministic transition matrix
    # ---

    # initialise determinist transitions matrix with all zeros
    deterministic_transitions = [
        [0] * nbr_poss_actions for _ in range(nbr_poss_actions)
    ]

    # for this previous-actions matrix, get the next-actions matrix
    for prev_actions, prev_ID in actions_2_ID.items():
        next_actions = [[None] * n for _ in range(n)]

        # first-author decisions
        # ---

        # first authors are unconstrained in their actions
        for focal in range(n):
            if action_gains_positive[prev_ID][focal][focal]:
                # if first-authorship has a positive unilateral switching gain,
                # then first-authorship will be pursued;
                next_act = 1

            else:
                # if first-authorship doesn't have a positive switching gain,
                # the decision depends on the focal's previous action
                # and first-authorship strategy
                next_act = calc_next_fsauthor_act(
                    prev_actions[focal][
                        focal
                    ],  # if focal first-authored (1) or not (0)
                    fsauthor_strats[focal],  # string used by the next-act calculator
                )

            # store next-act of focal first-author to other in the next-actions matrix
            next_actions[focal][focal] = next_act

        # diagonal of next-actions matrix is now populated

        # co-author decisions
        # ---

        # note: coauthors only update their action if the other will publish,
        # which is why this is calculated second
        for other in range(n):
            other_will_publish = next_actions[other][other] == 1

            # get the next act of focal to other
            for focal in range(n):
                if focal != other:
                    # already populated focal == other cases in next-action matrix

                    if not other_will_publish:
                        # NOTE: move this part outside the loop to improve efficiency
                        # if the other is not publishing, then focal's coauthorship
                        # strategy cannot not change -- represents "memory" of last act
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
                            next_act = calc_next_coauthor_act(
                                prev_ff, prev_fo, prev_of, prev_oo, coauthor_strat
                            )

                    # store next-act of focal to other in the next-actions matrix
                    next_actions[focal][other] = next_act

        # next-actions matrix is now fully populated

        # identify which actions index this next-actions matrix corresponds to
        next_actions = tuple(tuple(inner_list) for inner_list in next_actions)
        next_ID = actions_2_ID[next_actions]

        # store the transition from previous matrix ID to next matrix ID
        deterministic_transitions[prev_ID][next_ID] = 1

    return deterministic_transitions


# error transitions
# ===

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

def list_valid_transitions(fsco_actions_2_ID):
    # NOTE: this matrix is universal to all n-player games
    #
    # example: fsco_actions_2_ID for 2-player game
    # {
    #   ((0, 0), ((0,), (0,))): 0,
    #   ((0, 1), ((0,), (0,))): 1,
    #   ((0, 0), ((1,), (0,))): 2,
    #   ...
    #   ((1, 1), ((1,), (1,))): 15
    # }


    # format [(predecessor_1, successor_1), (predecessor_2, successor_2), ...]
    valid_transitions = list()

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

            if pre_coactionsV_fs0 == suc_coactionsV_fs0:
                valid_transitions.append((pre_ID, suc_ID))

    return valid_transitions

def calc_valid_transitions(fsco_actions_2_ID):
    # NOTE: this matrix is universal to all n-player games
    #
    # example: fsco_actions_2_ID for 2-player game
    # {
    #   ((0, 0), ((0,), (0,))): 0,
    #   ((0, 1), ((0,), (0,))): 1,
    #   ((0, 0), ((1,), (0,))): 2,
    #   ...
    #   ((1, 1), ((1,), (1,))): 15
    # }


    nbr_poss_actions = len(fsco_actions_2_ID)
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

    return valid_transitions

# symbolic stationary distribution
# ===


def symbolic_transitions_with_errors(pre_2_det, transition_errors_file_name):
    # Returns a matrix of symbolic transition probabilities.
    #
    #
    # Inputs:
    # ---
    #
    # pre_2_det, dict (int -> int)
    #   Each predecessor (key) and its deterministic successor (value).
    #
    # transition_errors_file_name, str
    #   File name for a CSV that contains information about the relationship 
    #   from and to every possible state, including whether a transition 
    #   between the two states ("from" is predecessor, "to" is ultimate successor) 
    #   is valid, and the number of erroneous and correct implementations 
    #   needed to perform the transition (e.g., if the "from" state is a 
    #   deterministic successor, and the "to" state is the ultimate successor). 
    #
    #   The CSV file should have the following columns:
    #   - "from_state": integer ID of state
    #   - "to_state": integer ID of state
    #   - "is_valid": boolean
    #   - "nbr_errors": integer
    #   - "nbr_correct": integer

    # read in the 2-player transitions and nbr errors
    # ---

    # creating the dictionary pre_2_suc, which is a nested dict 
    # with the structure:
    #       {
    #           from_state_1: {
    #               to_state_1: {
    #                   "is_valid": boolean, 
    #                   "nbr_errors_correct": (nbr_errors, nbr_correct)
    #               },
    #               to_state_2: {...},
    #               ...
    #           },
    #           from_state_2: {...},
    #           ...
    #       }

    df = pd.read_csv(transition_errors_file_name)
    pre_2_suc = {pre: dict() for pre in df["from_state"]}
    for pre, suc, is_valid, nbr_errors, nbr_correct in zip(
        df["from_state"],
        df["to_state"],
        df["is_valid"],
        df["nbr_errors"],
        df["nbr_correct"],
    ):
        pre_2_suc[pre][suc] = {
            "is_valid": is_valid,
            "nbr_errors_correct": (nbr_errors, nbr_correct),
        }


    # epsilon is the action-error probability
    eps = sp.symbols("eps")

    # initialise transition matrix with with all zeros
    nbr_states = len(pre_2_det)
    P = [[sp.Integer(0)] * nbr_states for _ in range(nbr_states)]

    for pre, det in pre_2_det.items():
        # valid successors from predecessor, ultimate successors
        ults = [ult for ult in pre_2_suc[pre].keys() if pre_2_suc[pre][ult]["is_valid"]]

        for ult in ults:
            # nbr of errors and correct implementations from the deterministic
            # successor to the ultimate successor
            nbr_errors, nbr_correct = pre_2_suc[det][ult]["nbr_errors_correct"]

            # put in equation for P
            P[pre][ult] = eps**nbr_errors * (1 - eps) ** nbr_correct

    return (np.array(P), eps)


def find_stationary_distribution(P, eps, max_pwr=2):

    nbr_states = P.shape[0]

    # matrix of epsilon coefficients, k_ij
    # where rows i correspond to the state
    # and columns j correspond to the power of epsiol
    coeffs = [
        [sp.symbols(f"k{state_idx}{pwr}") for pwr in range(max_pwr + 1)]
        for state_idx in range(nbr_states)
    ]

    # the stationary distribution expressed as a polynomial in epsilon
    # v_i = k_i0 eps^0 + k_i1 eps^1 + ...
    v = np.array(
        [
            sum(coeffs[state_idx][pwr] * eps**pwr for pwr in range(max_pwr + 1))
            for state_idx in range(nbr_states)
        ]
    )

    # at the stationary distribution, v = vP
    rhsV = list(v @ P)

    # At the stationary distribution, v = rhsV.
    # As the error eps -> 0, the stationary distribution approaches
    # the epsilon-order 0 coefficients, i.e.,
    #   v -> (k_10, k_20, k_30, ...)

    # solve by matching epsilon-power terms

    # first, from normalisation of the stationary distribution,
    # we always have: 1 = k_10 + k_20 + k_30 + ...
    pwr_2_eq0s = {0: [1 - sum(coeffs[state_idx][0] for state_idx in range(nbr_states))]}

    # then, from v = vP, each state has an equation involving various
    # epsilon-order terms:
    #   k_i0 + k_i1 eps + ... = [term]_i0 + [term]_i1 eps + ...
    for state_idx, rhs in enumerate(rhsV):
        pwrs_terms = rhs.as_poly(eps).all_terms()
        for pwr_tuple, term in pwrs_terms:
            pwr = pwr_tuple[0]
            if pwr <= max_pwr:
                eq0 = term - coeffs[state_idx][pwr]
                if eq0 != 0:  # exclude not-useful 0 = 0 equations
                    pwr_2_eq0s.setdefault(pwr, []).append(eq0)

    # we only want to solve the values of k_10, k_20, ...
    wants = [coeffs[state_idx][0] for state_idx in range(nbr_states)]

    # so we match epsilon terms, increasing the epsilon power
    # incrementally, until we have taken into account enough rare
    # sequences of errors that we can obtain the solutions we want
    stationary_distn = list()
    eq0s = list()
    for pwr_level in range(max_pwr + 1):
        # add to our system of equations
        # coefficient-matching at current power level
        eq0s += pwr_2_eq0s[pwr_level]

        # solve system of equations for epsilon coefficients
        free_coeffs = set.union(*[eq0.free_symbols for eq0 in eq0s])
        soln = sp.solve(eq0s, free_coeffs)  # returns [] if can't

        # check solution
        if all(want in soln for want in wants):
            # got something for each coeff we want
            stationary_distn_temp = [soln[want] for want in wants]
            if all([not propn.free_symbols for propn in stationary_distn_temp]):
                # each something we got had no unknown variables
                stationary_distn = stationary_distn_temp
                break

    return (stationary_distn, pwr_level)



# plotting deterministic graphs
# ===

def create_attributes_deterministic_graph(G, ID_2_actions, ID_2_fillcolor=None):

    # decorative attributes of each node for the output graph figure
    # ---

    attrs = {ID: dict() for ID in ID_2_actions}

    # give each node a tex label that is the action matrix
    for ID, actions in ID_2_actions.items():
        action_strs = [[str(v) for v in vv] for vv in actions]
        matrix_str = r" \\ ".join(
            [" & ".join(action_str) for action_str in action_strs]
        )
        texlbl = r"$\begin{pmatrix}" + matrix_str + r"\end{pmatrix}$"
        attrs[ID]["texlbl"] = texlbl

    # fillcolor each node according to the scheme given
    if ID_2_fillcolor is None:
        # default to nice blue
        attrs[ID]["fillcolor"] = "#bbccee"
    else:
        for ID, fillcolor in ID_2_fillcolor.items():
            attrs[ID]["fillcolor"] = fillcolor


    nx.set_node_attributes(G, attrs)

    # decorative attributes of each edge for the output graph figure
    # ---

    attrs = {
        (u, v): {
            "lblstyle": "auto, sloped, black, draw = none"
        }
        for u, v in G.edges()
    }

    nx.set_edge_attributes(G, attrs)

    # create the pygraphviz graph with attributes
    # ---

    # make attribute graph with particular attributes

    AG = nx.nx_agraph.to_agraph(G)

    # global attributes
    AG.graph_attr["overlap"] = "false"
    AG.graph_attr["d2tgraphstyle"] = "every node/.style={draw}"
    AG.node_attr["style"] = "filled"
    AG.node_attr["shape"] = "circle"
    AG.node_attr["color"] = "white"

    return AG

def print_attributes_deterministic_graph(
    AG, 
    prefix_out=None, 
    layout_alg="neato"
):
    # if we're just running to check something
    if prefix_out is None:
        prefix_out = "spare"

    # create dot file
    AG.write(f"{prefix_out}.dot")  # write to dot file

    # use dot2tex to convert: .dot -> .tex
    subprocess.run(
        f"dot2tex --prog={layout_alg} --format=tikz --figonly --tikzedgelabel {prefix_out}.dot > {prefix_out}.tex",
        shell=True,
    )

    # write stand-alone file that will import .tex file above
    standalone_txt = """
    \\documentclass{standalone}
    \\usepackage[x11names, svgnames, rgb]{xcolor}
    \\usepackage[utf8]{inputenc}
    \\usepackage{tikz}
    \\usetikzlibrary{snakes,arrows,shapes}
    \\usepackage{amsmath}
    \\begin{document}
    """
    standalone_txt += "\\input{"
    standalone_txt += f"{prefix_out}"
    standalone_txt += ".tex}\n\\end{document}"

    with open("standalone_temp.tex", "w") as f:
        f.write(standalone_txt)

    # compile the standalone file to make a pdf of the graph
    subprocess.run(f"pdflatex -jobname={prefix_out} standalone_temp.tex", shell=True)
