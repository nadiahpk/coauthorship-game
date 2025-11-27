# Create a CSV file that lists every from state, to state, whether the transition is valid,
# the number of errors between the two transitions, and the number of correct actions.
# The validity is about ensuring that errors are not applied to coauthorship memories,
# and the number of errors and correct actions is used to calculate the probability of the 
# transition from a deterministic to ultimate successor.

import os
import model_fncs as mf
from pathlib import Path
import pandas as pd


# parameters
# ---

# directory to save results
results_dir = Path(os.environ.get('RESULTS', "")) / "deterministic"

# number of players in the game
n = 2


# create list of valid transitions
# ---

ID_2_actions = mf.get_ID_2_actions(n)
fsco_actions_2_ID = {
    mf.actions_2_fsactions_coactions(actions): ID for ID, actions in ID_2_actions.items()
}
valid_transitionsV = mf.list_valid_transitions(fsco_actions_2_ID)


# count the number of errors to get from each predecessor to successor
# ---

ID_2_fsco_actions = {ID: fsco_actions for fsco_actions, ID in fsco_actions_2_ID.items()}
nbr_possible_actions = len(ID_2_fsco_actions)

# format: [(from_state_1, to_state_1, is_valid_1, nbr_errors_1, nbr_correct_1), (from_state_2, ...), ...]
df_list = list()
for suc_ID in range(nbr_possible_actions):
    # the ultimate predecessor in a convenient form
    suc_fsactionV, suc_coactionsV = ID_2_fsco_actions[suc_ID]

    # indices where first author is authoring
    fs1_idxs = [idx for idx, fsaction in enumerate(suc_fsactionV) if fsaction == 1]

    for pre_ID in range(nbr_possible_actions):

        # the comparison is made to the predecessor
        pre_fsactionV, pre_coactionsV = ID_2_fsco_actions[pre_ID]

        # count all first-author differences
        nbr_fs_errors = sum(
            suc_fsaction != pre_fsaction
            for suc_fsaction, pre_fsaction in zip(suc_fsactionV, pre_fsactionV)
        )

        # count only those co-author differences that occur where the 
        # first author is authoring in the ultimate successor


        # count only those rows (co-authorship relationships)
        nbr_co_errors = sum(
            sum(
                suc_coaction != pre_coaction
                for suc_coaction, pre_coaction in zip(suc_coactionsV[fs1_idx], pre_coactionsV[fs1_idx])
            )
            for fs1_idx in fs1_idxs
        )

        # get the number of errors, number correct, and store
        nbr_errors = nbr_fs_errors + nbr_co_errors
        nbr_correct = n + len(fs1_idxs) * (n - 1) - nbr_errors

        # is this a valid transition?
        is_valid = (pre_ID, suc_ID) in valid_transitionsV

        # store
        df_list.append((pre_ID, suc_ID, is_valid, nbr_errors, nbr_correct))


# save to CSV file
# ---

fname = f"transitions_errors_nbr_players_{n}.csv"
path_out = results_dir / fname
df = pd.DataFrame(
    df_list,
    columns = ["from_state", "to_state", "is_valid", "nbr_errors", "nbr_correct"] # type: ignore
)
df.to_csv(path_out, index=False)
 

# save to parquet as well (trying new things)
# ---

df.to_parquet(results_dir / f"transitions_errors_nbr_players_{n}.parquet", index=False)
