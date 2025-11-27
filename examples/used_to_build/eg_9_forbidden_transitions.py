# figure out how to calculate the valid transitions and the number of errors
#
# this code got used in eg_8

import itertools as it


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


# ----

# nbr players
n = 3

# create an arbitrary predecessor action-state
pre = [[1, 0, 1], [1, 0, 0], [1, 1, 0]]

# create an arbitrary determinstic successor action-state
det = [[0, 1, 1], [1, 1, 0], [1, 0, 0]]
# players 0 and 2 are not first-authoring,
# so memories of their coauthors must be preserved

# here is a potential successor state that I believe can be reached
# because only the coauthorship relationships with player 1 are changed
# 2 errors
can = [[0, 0, 1], [1, 1, 0], [1, 1, 0]]

# this is not possible because Player 1 is no longer first-authoring
# yet Player 0's coauthorship with Player 1 is changed
nop = [[0, 1, 1], [1, 0, 0], [1, 1, 0]]

# this is possible because Player 1's coauthors don't change
can2 = [[0, 0, 1], [1, 0, 0], [1, 1, 0]]

# here's another valid transition
can3 = [[0, 0, 0], [1, 1, 1], [1, 0, 1]]

error_actions = [can, nop, can2]


# how to test if transition is forbidden?
# ===

print(f"Predecessor: {pre}")

error_actionsV = [can, nop, can2, can3]
for err in error_actionsV:
    # let's start by getting just those elements I need to compare

    # the set of all first authors in the determistic successor who are not publishing
    non_fsauthors = [i for i in range(n) if err[i][i] == 0]
    # -- [0, 1, 2]

    # the coauthor memories are kept in the columns excluding the first-author themselves
    err_memorys = [
        [err[i][non_fsauthor] for i in range(n) if i != non_fsauthor]
        for non_fsauthor in non_fsauthors
    ]
    # it's the columns transposed
    # -- [[1, 1], [1, 1], [1, 0]]

    # they need to match the predecessor states
    pre_memorys = [
        [pre[i][non_fsauthor] for i in range(n) if i != non_fsauthor]
        for non_fsauthor in non_fsauthors
    ]
    # -- [[1, 1], [0, 1], [1, 0]]

    if err_memorys == pre_memorys:
        print(f"Valid transition: {err}")
    else:
        print(f"Invalid transition: {err}")


# use the fsactionsV, coactionsV to check validity of transitions
# ===

# predecessor
print("\n---\n")
# print(f"Predecessor in action-matrix form: {pre}")

pre_fsactionV, pre_coactionsV = actions_2_fsactions_coactions(pre)
print(f"Predecessor first-author actions: {pre_fsactionV}")
print(f"Predecessor co-author actions: {pre_coactionsV}")

det_fsactionV, det_coactionsV = actions_2_fsactions_coactions(det)

# deterministic successor
print("\n---\n")

# NOTE: this matrix, with valid and invalid transitions, is universal
# the calculation of number of errors depends on the deterministic successor

ult_actionsV = [can, nop, can2, can3]
for ult_actions in ult_actionsV:
    ult_fsactionV, ult_coactionsV = actions_2_fsactions_coactions(ult_actions)

    # need to check that co-author actions of non-first-authoring match predecessor
    fs0_idxs = [idx for idx, fsaction in enumerate(ult_fsactionV) if fsaction == 0]
    pre_coactionsV_fs0 = [pre_coactionsV[fs0_idx] for fs0_idx in fs0_idxs]
    ult_coactionsV_fs0 = [ult_coactionsV[fs0_idx] for fs0_idx in fs0_idxs]

    if pre_coactionsV_fs0 != ult_coactionsV_fs0:
        print("\nInvalid transition")
    else:
        print("\nValid transition:")
        print(f"Predecessor first-author actions:             {pre_fsactionV}")
        print(f"Deterministic successor first-author actions: {det_fsactionV}")
        print(f"Ultimate successor first-author actions:      {ult_fsactionV}")
        print(f"Predecessor co-author actions:             {pre_coactionsV}")
        print(f"Deterministic successor co-author actions: {det_coactionsV}")
        print(f"Ultimate successor co-author actions:      {ult_coactionsV}")

        # count the number of errors required to reach this ultimate successor
        fs1_idxs = [idx for idx, fsaction in enumerate(ult_fsactionV) if fsaction == 1]

        # first-author errors
        nbr_fs_errors = sum(
            ult_fsaction != det_fsaction
            for ult_fsaction, det_fsaction in zip(ult_fsactionV, det_fsactionV)
        )

        # co-author errors
        nbr_co_errors = sum(
            sum(
                ult_coaction != det_coaction
                for ult_coaction, det_coaction in zip(ult_coactionsV[fs1_idx], det_coactionsV[fs1_idx])
            )
            for fs1_idx in fs1_idxs
        )
        
        nbr_errors = nbr_fs_errors + nbr_co_errors
        nbr_correct = n + len(fs1_idxs) * (n - 1) - nbr_errors

        print(f"nbr. errors: {nbr_errors}")
        print(f"nbr. correct: {nbr_correct}")