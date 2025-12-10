Notes to self:
- generating the valid transitions and with errors can be put in a separate results

- `eg_7_determinstic.py`
    - generate and plot deterministic transitions
    - uses function `calc_deterministic_transitions.py`

- `eg_11_count_all_errors.py`
    - produced `transitions_errors_nbr_players_2.csv`, which is generic. 
        - only needs the number of players, 
        and it can identify what is a valid transition and the number of errors/correct between pairs

- `eg_12_solve_stat_dist.py`
    - hardcoded in the deterministic transitions
        - used to create dictionary `pre_2_det`
    - read in `transitions_errors_nbr_players_2.csv`
    - got symbolic transition matrix with error terms
        - function used is `symbolic_transitions_with_errors()`
        - inputs were `pre_2_det` and `transitions_errors_nbr_players_2.csv`
        - outputs were `P` and `eps`
    - found the stationary distribution
        - function used was `find_stationary_distribution`
        - inputs were `P` and `eps`
        - outputs were `stationary_distn` and `pwr_level`
