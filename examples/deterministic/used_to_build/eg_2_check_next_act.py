# Do the introspection dynamics for two players from Simple Model 1
#
# -- I can get the next actions as first author and co-author out

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
            sum(actions[i][first_author] for i in range(n))
            - actions[first_author][first_author]
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


def calc_next_coauthor_act(prev_act, strategy_name):
    # Calculate each co-authorship next action based on
    # previous actions and the coauthorship strategy name

    if strategy_name == "pavlov":
        if (prev_act == (0, 0)) or (prev_act == (1, 1)):
            next_act = 1
        elif (prev_act == (1, 0)) or (prev_act == (0, 1)):
            next_act = 0
        else:
            raise ValueError(
                f"Invalid prev_act: '{prev_act}'. Expected (0, 0), (0, 1), (1, 0), or (1, 1)."
            )
    elif strategy_name == "all_c":
        next_act = 1
    elif strategy_name == "all_d":
        next_act = 0
    else:
        raise ValueError(
            f"Invalid strategy_name: '{strategy_name}'. Expected 'pavlov' or 'all_c' or 'all_d'."
        )

    return next_act


def calc_next_fsauthor_act(prev_act, strategy_name):
    # Calculate each first-authorship next action based on
    # previous actions and the first-author strategy name

    if strategy_name == "always":
        next_act = 1
    else:
        raise ValueError(
            f"Invalid strategy_name: '{strategy_name}'. Expected 'always'..."
        )

    return next_act


# --------------------------------

# parameters
# ---

n = 3  # number of players
b = 2  # maximum benefit of a paper
c_f = 1  # cost of first-authoring
c_c = 0.8  # cost of co-authoring

# topic preferences
preferences = np.array(
    [
        np.array([0]),
        np.array([1]),
        np.array([3.5]),
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
    "possible_coauthor_strats": ["all_c", "all_d", "pavlov"],
    "possible_fsauthor_strats": ["always"],
}


# initialise
# ---

coauthor_strats = ["pavlov", "pavlov", "pavlov"]
coauthor_strats = ["all_c", "all_d", "pavlov"]
fsauthor_strats = ["always", "always", "always"]


# rows are co-author, columns are first-author
prev_actions = [
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]


# author actions
# ---

print("Previous actions:")
print(prev_actions)


# the strategy is a function of the pairwise last action
n = params["nbr_players"]
next_actions = [
    [
        calc_next_fsauthor_act(prev_actions[focal][focal], fsauthor_strats[focal])
        if focal == other
        else calc_next_coauthor_act(
            (prev_actions[focal][other], prev_actions[other][focal]),
            coauthor_strats[focal],
        )
        for other in range(n)
    ]
    for focal in range(n)
]
print("Next actions:")
print(next_actions)