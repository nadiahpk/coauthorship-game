# put the functions used in the examples for here for now
#
# updates:
# 1. calc_next_coauthor_act()
#   - pavlov updated to have a fallback strategy
#   - the coauthorship strategy updated to also accept focal and
#     other first-authorship actions

from scipy.spatial import distance

import networkx as nx
import subprocess


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

    # check we know the strategy name (in loop for checking own errors)
    valid_strategy_names = ["pavlov-c", "pavlov-d", "all_c", "all_d"]

    # if other did not publish, should not update coauthorship (no information with which to do so)
    if prev_oo == 0:
        raise ValueError(
            "Should not be using co-authorship update rules when the other is not publishing."
        )

    # calculate next act based on co-authorship strategy
    # ---

    if strategy_name == "all_c":
        next_act = 1

    elif strategy_name == "all_d":
        next_act = 0

    elif (strategy_name == "pavlov-c") or (strategy_name == "pavlov-d"):
        if prev_ff == 0:
            # Pavlov is a reciprocal strategy and so only applies
            # when both players first-authored a paper that the other
            # could choose to cooperate or defect on. Therefore,
            # in this situation, we apply the fallback rule
            if strategy_name == "pavlov-c":
                next_act = 1
            else:  # strategy_name == "pavlov-d":
                next_act = 0

        elif ((prev_fo, prev_of) == (0, 0)) or ((prev_fo, prev_of) == (1, 1)):
            next_act = 1

        elif ((prev_fo, prev_of) == (1, 0)) or ((prev_fo, prev_of) == (0, 1)):
            next_act = 0

        else:
            raise ValueError(
                f"Invalid previous actions: {prev_ff}, {prev_fo}, {prev_of}, {prev_oo}."
            )

    else:  # strategy_name not in valid_strategy_names:
        raise ValueError(
            f"Invalid strategy_name: '{strategy_name}'. Expected one of: {', '.join(valid_strategy_names)}."
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


def calc_action_gains(
    params,
    actions,
    gammas=None,
    alignments=None,
):
    # Calculate the direct payoff from unilaterally switching to author
    # to not-author for each player. A positive action gain means
    # that the focal player should author (first- or co-author) that
    # paper regardless of any strategic considerations (e.g., tit-for-tat)
    # because the payoff from authoring exceeds the cost

    # extract needed parameters
    # ---

    n = params["nbr_players"]
    preferences = params["preferences"]

    # calculate optional components if not given
    # ---

    if gammas is None:
        gammas = [[gamma_fnc(params, i, j) for j in range(n)] for i in range(n)]

    if alignments is None:
        topics = topics_fnc(params)
        alignments = alignments_fnc(params, preferences, topics)

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


# plotting deterministic transitions graphs


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

    # create the pygraphviz graph with attributes
    # ---

    # make attribute graph with particular attributes
    nx.set_node_attributes(G, attrs)
    AG = nx.nx_agraph.to_agraph(G)

    # global attributes
    AG.graph_attr["overlap"] = "false"
    AG.graph_attr["d2tgraphstyle"] = "every node/.style={draw}"
    AG.node_attr["style"] = "filled"
    AG.node_attr["shape"] = "circle"
    AG.node_attr["color"] = "white"

    return AG


def print_attributes_deterministic_graph(AG, prefix_out=None):
    # if we're just running to check something
    if prefix_out is None:
        prefix_out = "spare"

    # create dot file
    AG.write(f"{prefix_out}.dot")  # write to dot file

    # use dot2tex to convert: .dot -> .tex
    subprocess.run(
        f"dot2tex --prog=neato --format=tikz --figonly {prefix_out}.dot > {prefix_out}.tex",
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
