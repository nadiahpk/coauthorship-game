import networkx as nx
import subprocess
import numpy as np

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

# --------------------

# parameters and previous steps
# ---

prefix_out = "eg_5"

n = 2

ID_2_actions = {
    0: ((0, 0), (0, 0)),
    1: ((0, 0), (0, 1)),
    2: ((0, 0), (1, 0)),
    3: ((0, 0), (1, 1)),
    4: ((0, 1), (0, 0)),
    5: ((0, 1), (0, 1)),
    6: ((0, 1), (1, 0)),
    7: ((0, 1), (1, 1)),
    8: ((1, 0), (0, 0)),
    9: ((1, 0), (0, 1)),
    10: ((1, 0), (1, 0)),
    11: ((1, 0), (1, 1)),
    12: ((1, 1), (0, 0)),
    13: ((1, 1), (0, 1)),
    14: ((1, 1), (1, 0)),
    15: ((1, 1), (1, 1)),
}

deterministic_transitions = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
]


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


# create attributes graph and print
# ---

AG = create_attributes_deterministic_graph(G, ID_2_actions, ID_2_fillcolor)
print_attributes_deterministic_graph(AG, prefix_out)