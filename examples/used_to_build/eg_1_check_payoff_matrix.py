# Do the introspection dynamics for two players from Simple Model 1
#
# Checking my calculations of the payoff matrix against what I did in the pdf
# -- agree

from scipy.spatial import distance
import numpy as np


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
    influences = [0] * n
    for first_author in range(n):
        nbr_coauthors = (
            sum(actions[i][first_author] for i in range(n)) - actions[first_author][first_author]
        )
        influences[first_author] = b * (wf + nbr_coauthors) / (wf + n - 1)

    return influences


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
                    1 / (
                        1 + distance.euclidean(
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


# --------------------------------

# parameters
# ---

# only consider two strategies: all-D and Pavlov
strat2name = {
    "allD": "Always defect",
    "pavl": "Pavlov",
}

# number of players
n = 2

# maximum benefit of a paper
b = 2.8

c_f = 0.9  # cost of first-authoring
c_c = 0.5  # cost of co-authoring

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
    "strat2name": strat2name,
    "nbr_players": n,
    "preferences": preferences,
    "type_of_good": "excludable",
    "gamma_rule": "constant_1",
    # benefit of papers
    "first_author_weight": 1,  # the weighting of the first author towards the paper's influence
    "max_paper_benefit": b,
    # alignment between preference and topic
    "distance_measure": "euclidean",
    # authoring costs
    "authoring_cost_rule": "simple",
    "first_author_cost": c_f,
    "coauthor_cost": c_c,
    # topic-choice
    "topic_choice_rule": "exact_preference",
}


# check I can calculate basic things
# ---

# list out each of the actions possible in the simple model
# keys refer to the location in the payoff matrix
possible_actions = {
    "a": [[1, 1], [1, 1]],
    "b": [[1, 1], [0, 1]],
    "c": [[1, 0], [1, 1]],
    "d": [[1, 0], [0, 1]],
}

print("preferences = ")
print(preferences)

# calculate topics chosen
topics = topics_fnc(params)
print("topics = ")
print(topics)

# calculate alignments between co-authors' preferences (rows) and first-authors' topics (columns)
alignments = alignments_fnc(params, preferences, topics)
print("alignments = ")
print(alignments)


# calculate payoffs
actions = possible_actions["a"]
print("I have chosen actions (row is coauthor, col is first_author):")
print(actions)

# calculate the influence of the paper each writes
influences = papers_influences_fnc(params, actions)

print("paper influences = ")
print(influences)


# calculate by hand, based on my pdf, the payoff to each player for each action
# ---
print("---")

# because each first author chooses topics that exactly match their preference,
# the alignment between first-author and co-author is
distance = distance.euclidean(preferences[0], topics[1])
print(f"Distance between preferences = {distance}")
A = 1 / (1 + distance)
print(f"A = {A}")

# costs of first-authoring and co-authoring
cf = params["first_author_cost"]
cc = params["coauthor_cost"]
print(f"Authoring costs are: cf = {cf}, cc = {cc}")

# coefficient for influence of the paper
b2 = params["max_paper_benefit"] / 2
print(f"b2 = {b2}")

print("By hand, I calculate payoff-matrix elements:")
#preferences = np.array(
# payoff-matrix element a
pm_a = b2 * (2 + 2*A) - cf - cc
pm_b = b2 * (1 + 2*A) - cf - cc
pm_c = 2 * b2 - cf
pm_d = b2 - cf

print(f" a = {pm_a}")
print(f" b = {pm_b}")
print(f" c = {pm_c}")
print(f" d = {pm_d}")

# look closer at a (1, 1)
# pay = b2 * (1 + 1 + ai * (1 + A)) - cf - cc

# ---

print("Using my code, I calculate payoff-matrix elements:")

gammas = [
    [
        gamma_fnc(params, i, j)
        for j in range(n)
    ]
    for i in range(n)
]

for element, actions in possible_actions.items():
    i = 0  # first author
    influences = papers_influences_fnc(params, actions)
    pay = sum(
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
    print(f" {element} = {pay}")



