# functions for plotting graphs
import networkx as nx
import subprocess

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
        for ID in ID_2_actions:
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

