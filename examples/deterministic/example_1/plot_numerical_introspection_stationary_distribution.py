# Plot the change in the proportion of time the introspection 
# dynamics spends in each strategy-pair state as a function 
# of the introspection-strength parameter delta
#
# Note that this only looks at dynamics of co-author strategy,
# and first-author strategy is assumed fixed ("never").
#
# Take advantage of the symmetry between players by grouping 
# strategy pairs, e.g., sum the time spent in (pavlov_c, pavlov_d)
# with (pavlov_d, pavlov_c) into one probability 
# {pavlov_c, pavlov_d}.

# import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
import itertools as it

import matplotlib.pyplot as plt
import tol_colors as tc

#import model_fncs as mf


# parameters
# ---

# where the introspection-dynamics stationary distributions are stored
path_in = "numerical_introspection_stationary_distribution.parquet"

# where to store plot
path_out = "numerical_introspection_stationary_distribution.pdf"


# read in stationary distributions 
# ---

table = pq.read_table(path_in)

# the first column is called "introspection_strength", and all other 
# columns are strategy pairs, e.g., pavlov_c_Vs_pavlov_d


# merge reversed strategy pairs
# ---

strats_cols = [col_name for col_name in table.column_names if col_name != "introspection_strength"]
strat_uniques = {col_name.split("_Vs_")[0] for col_name in strats_cols}
strats_2_cols = {
        strats: list({f"{strats[0]}_Vs_{strats[1]}", f"{strats[1]}_Vs_{strats[0]}"})
        for strats in it.combinations_with_replacement(strat_uniques, 2)
}

strats_2_stationarys = {
        strats: sum(table[col].to_numpy() for col in cols)
        for strats, cols in strats_2_cols.items()
}
nbr_strats = len(strats_2_stationarys)


# plot 
# ---

# NOTE: I bet this would be nicer with ultraplot 
plt.figure(figsize=(5, 4))

deltas = table["introspection_strength"].to_pylist()

cmap = tc.rainbow_discrete(10)
for idx, (strats, stationarys) in enumerate(strats_2_stationarys.items()):
    plt.plot(deltas, stationarys, color=cmap(idx), alpha=0.9, lw=2, label = ", ".join(strats))

plt.legend(loc="lower right", fontsize="x-small")
plt.xlabel(r"introspection strength, $\delta$", fontsize="large")
plt.ylabel("probability", fontsize="large")
plt.xlim((0, 40))
plt.tight_layout()
plt.savefig(path_out)
plt.close("all")

"""

# create useful dictionaries 
# ---

# read in the expected payoff to each strategy in each strategy-pair scenario 
table = pq.read_table(
        path_in,
        columns = [
            "player_1_coauthor_strategy", 
            "player_2_coauthor_strategy", 
            "player_1_payoff", 
            "player_2_payoff", 
        ]
)

# use table to make dictionary from strategy pair to payoff pair 
strats_2_pays = {
        (row["player_1_coauthor_strategy"], row["player_2_coauthor_strategy"]):
        (row["player_1_payoff"], row["player_2_payoff"])
        for row in table.to_pylist()
}
# note this dictionary takes advantage of the symmetry between players,
# listing only combinations-with-replacement pairs of strategies 
# rather than the full product of strategy pairs

# dictionaries indexing the coauthor strategies 
ordered_strats = sorted(set(
    table["player_1_coauthor_strategy"].to_pylist() 
    + table["player_1_coauthor_strategy"].to_pylist()
))
nbr_strats = len(ordered_strats)
strat_idx_2_strat = dict(enumerate(ordered_strats))
strat_2_strat_idx = {strat: idx for idx, strat in strat_idx_2_strat.items()}

# indexation of the transition matrix 
mat_idx_2_strat_idxs = dict(enumerate(list(it.product(strat_idx_2_strat.keys(), repeat=2))))
strat_idxs_2_mat_idx = {strat_idxs: mat_idx for mat_idx, strat_idxs in mat_idx_2_strat_idxs.items()}

# a dictionary mapping from coauthor-strategy index pair to payoffs pair
# this dictionary contains the full product
mat_idx_2_pays = dict()
for mat_idx, strat_idxs in mat_idx_2_strat_idxs.items():
    strats = (strat_idx_2_strat[strat_idxs[0]], strat_idx_2_strat[strat_idxs[1]])
    if strats in strats_2_pays:
        mat_idx_2_pays[mat_idx] = strats_2_pays[strats]
    else:
        # the strategy payoffs were stored taking advantage of symmetry between players
        pays_rev = strats_2_pays[(strats[1], strats[0])]
        mat_idx_2_pays[mat_idx] = (pays_rev[1], pays_rev[0])


# identify the permissible transitions for players 1 and 2
# ---

# players only update their strategy one at a time, so the only 
# permissible transitions from (strat_player_1, strat_player_2) = (k, l)
# are (k, l) -> (k', l) and (k, l) -> (k, l')

# size of the transition matrix
mat_len = len(mat_idx_2_strat_idxs)

# storage: {old_mat_idx: [list of new_mat_idxs for p1, list of new_mat_idxs for p2]}
permissible_old_2_new_mat_idxs = {old_mat_idx: list() for old_mat_idx in range(mat_len)}

for old_mat_idx in range(mat_len):
    old_strat_idxs = mat_idx_2_strat_idxs[old_mat_idx]

    # permissible transitions for player 1
    p1_old_strat_idx = old_strat_idxs[0]
    p1_permissible_new_strat_idxs = [
        (new_strat_idx, old_strat_idxs[1]) for new_strat_idx in range(nbr_strats) 
        if new_strat_idx != p1_old_strat_idx
    ]
    p1_permissible_new_mat_idxs = [
            strat_idxs_2_mat_idx[new_strat_idxs] 
            for new_strat_idxs in p1_permissible_new_strat_idxs
    ]

    # permissible transitions for player 2
    p2_old_strat_idx = old_strat_idxs[1]
    p2_permissible_new_strat_idxs = [
        (old_strat_idxs[0], new_strat_idx) for new_strat_idx in range(nbr_strats) 
        if new_strat_idx != p2_old_strat_idx
    ]
    p2_permissible_new_mat_idxs = [
            strat_idxs_2_mat_idx[new_strat_idxs] 
            for new_strat_idxs in p2_permissible_new_strat_idxs
    ]

    permissible_old_2_new_mat_idxs[old_mat_idx] = [p1_permissible_new_mat_idxs, p2_permissible_new_mat_idxs]


# for each introspection-strength parameter value, delta, find 
# the stationary distribution of the introspection dynamics
# ---

# to store stationary distributions, corresponds to list of deltas
stationarys = list()

for delta in deltas:

    # create the introspection-dynamics markov-chain transition matrix

    # initialise the transition matrix
    mat = np.zeros((mat_len, mat_len))

    # populate the off-diagonal elements
    for old_mat_idx in range(mat_len):

        old_strat_idxs = mat_idx_2_strat_idxs[old_mat_idx]
        old_pays = mat_idx_2_pays[old_mat_idx]


        for focal_idx in [0, 1]:
            for new_mat_idx in permissible_old_2_new_mat_idxs[old_mat_idx][focal_idx]:
                new_pays = mat_idx_2_pays[new_mat_idx]
                mat[old_mat_idx, new_mat_idx] = (1 / 2) * (1 / (nbr_strats - 1)) * \
                        prob_transition_given_comparison_pair(delta, focal_idx, old_pays, new_pays)

    # populate the diagonal elements
    for mat_idx in range(mat_len):
        mat[mat_idx, mat_idx] = 1 - sum(mat[mat_idx, :])


    # find and store the stationary distribution of the introspection dynamics 

    eigenvals, eigenvecs = np.linalg.eig(mat.T) # transpose bc left eigenvecs
    i = np.argmin(np.abs(eigenvals - 1))
    stationary = np.real(eigenvecs[:, i])
    stationary = stationary / np.sum(stationary)
    stationarys.append(stationary.tolist())


# write to parquet
# ---

table_dict = {"introspection_strength": deltas}

# restructure the stationarys list so each stategy-pair is in its own column
# with rows corresponding to the delta value
stationarys_cols = list(zip(*stationarys))

# corresponding column names 
col_names = [
        strat_idx_2_strat[mat_idx_2_strat_idxs[mat_idx][0]] + "_Vs_" +
        strat_idx_2_strat[mat_idx_2_strat_idxs[mat_idx][1]]
        for mat_idx in range(mat_len)
]

# add each stationary distribution to the table dictionary
for mat_idx in range(mat_len):
    table_dict[col_names[mat_idx]] = list(stationarys_cols[mat_idx])

# build PyArrow table and save
table = pa.table(table_dict)
pq.write_table(table, path_out)

"""
