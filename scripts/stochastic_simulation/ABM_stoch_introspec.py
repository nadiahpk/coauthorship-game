import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

# Game payoffs (asymmetric with classic prisoner dilemma payoffs)
U1 = {
    (1,1): 4, (1,0): 0,
    (0,1): 5, (0,0): 1
}

U2 = {
    (1,1): 4, (0,1): 0,
    (1,0): 5, (0,0): 1
}

'''# Game payoffs (symmetric with slightly different payoff proportions)
U1 = {
    (1,1): 5, (1,0): 1,
    (0,1): 4, (0,0): 2
}

U2 = {
    (1,1): 5, (0,1): 4,
    (1,0): 1, (0,0): 2
}'''


#### Strategies
strategies = {
    "ALLC": np.array([1, 1, 1, 1]),
    "ALLD": np.array([0, 0, 0, 0]),
    "PAVLOV": np.array([1, 0, 0, 1]) # pavlov c and d the same in this context
}
def strat_name_from_vec(vec):
    for name, v in strategies.items():
        if np.array_equal(vec, v):
            return name
    raise ValueError("Unknown strategy vector")
joint_strats = list(itertools.product(strategies.keys(), repeat=2))
joint_index = {js: i for i, js in enumerate(joint_strats)}


# Game States
states = [(1,1), (1,0), (0,1), (0,0)]  # CC, CD, DC, DD
state_index = {s: i for i, s in enumerate(states)}



def equilibrium_time(prob_dist, eps, min_t, persistence):
    # checks if current prob dis is ~= previous prob dis and makes sure stable for no. of rounds declared in persistence. 
    # Epsilon is amount of reasonable noise allowed
    stable = 0

    for t in range(min_t, len(prob_dist) - 1):
        if np.linalg.norm(prob_dist[t+1] - prob_dist[t], ord=1) < eps:
            stable += 1
            if stable >= persistence:
                return t - persistence + 1
        else:
            stable = 0

    return None




def checkstrat (p1_strat, p2_strat):
    ### Compares strategies between two current players

    global strategies
    available_strats = list(strategies.keys())

    # pick learner
    learner = random.choice([1,2])
    # based on who is testing strat, determine pi_r and pi_l
    if learner == 1:
        other = 2
        learner_strat = p1_strat
        rolemodel_name = random.choice([s for s in available_strats if not np.array_equal(strategies[s], learner_strat)])
        rolemodel_strat = strategies[rolemodel_name]
        pi_r = markov_matrix(rolemodel_strat, p2_strat)
        pi_l = markov_matrix(p1_strat, p2_strat)
    else:
        other = 1
        learner_strat = p2_strat
        rolemodel_name = random.choice([s for s in available_strats if not np.array_equal(strategies[s], learner_strat)])
        rolemodel_strat = strategies[rolemodel_name]
        pi_r = markov_matrix(p1_strat, rolemodel_strat)
        pi_l = markov_matrix(p1_strat, p2_strat)

    # compute stationary distributions
    v_r = stationary_distribution(pi_r)
    v_l = stationary_distribution(pi_l)

    # expected payoffs
    if learner ==1:
        payoff_r = sum(v_r[i] * U1[states[i]] for i in range(4))
        payoff_l = sum(v_l[i] * U1[states[i]] for i in range(4))
    if learner == 2:
        payoff_r = sum(v_r[i] * U2[states[i]] for i in range(4))
        payoff_l = sum(v_l[i] * U2[states[i]] for i in range(4))


    # probability to switch
    p_switch = 1 / (1 + np.exp(-delta * (payoff_r - payoff_l)))
    if np.random.rand() < p_switch: # check range for rand num
        if learner == 1:
            p1_strat = rolemodel_strat
        else:
            p2_strat = rolemodel_strat


    ## This function:
    # learner = pick randomly p1 or 2 and holds the current strat they doing
    # rolemodel = a new, randomly picked, strat that is not the learners strat (out of the available strats)
    # otherp = the strat and actions of the other player that was not picked
    # pi_r = the transition matrix of rolemodel strategy IF the chosen player played role model with other players current strat
    # pi_l = the transition matrix of learner strategy IF chosen player continues learner strat with other players current strat
    # Payoff_r = v*A (where v is the eigenvector of pi_r)
    # Payoff_l = v*A (where v is the eigenvector of pi_l)
    # compute p_l = 1/(1+e^(-delta(pi_r-pi_l)))
    # if probability of p_l is less than random number 0-1, then change strat of learner player to rolemodel strat


    return p1_strat, p2_strat

def stationary_distribution(M):
    eigvals, eigvecs = np.linalg.eig(M.T)
    v = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    v = v[:,0]
    return v / v.sum()

### Build transition matrix from strategies
def markov_matrix(p1, p2):
    M = np.zeros((4,4))

    for i, (a1_prev, a2_prev) in enumerate(states):
        # Player 1: normal perspective
        p1c = p1[i]

        # Player 2: flipped perspective
        j = state_index[(a2_prev, a1_prev)]
        p2c = p2[j]

        M[i, state_index[(1,1)]] = p1c * p2c
        M[i, state_index[(1,0)]] = p1c * (1 - p2c)
        M[i, state_index[(0,1)]] = (1 - p1c) * p2c
        M[i, state_index[(0,0)]] = (1 - p1c) * (1 - p2c)


    return M



def play_game(rounds, epsilon, delta):
    # Starting Strategies of both players
    p1_strat = strategies["ALLD"]
    p2_strat = strategies["ALLC"]
    history_states = []
    # To record strategy history
    p1_strat_hist = []
    p2_strat_hist = []

    # Probability Distribution for joint strategies
    joint_counts = np.zeros(len(joint_strats))
    prob_dist_stratpairs = []

    for t in range(rounds):
        
        # record current joint strategies
        current_joint_strat = (
            strat_name_from_vec(p1_strat),
            strat_name_from_vec(p2_strat)
        )
        # calculate prob_dist for this round
        joint_counts[joint_index[current_joint_strat]] += 1
        prob_dist_stratpairs.append(joint_counts / (t + 1))

        # record current strategies
        p1_strat_hist.append(p1_strat.copy())
        p2_strat_hist.append(p2_strat.copy()) # Pair these together

        # Introspect at the end of each round
        p1_strat, p2_strat = checkstrat(p1_strat, p2_strat)

    return (p1_strat_hist,p2_strat_hist,np.array(prob_dist_stratpairs))


deltas = [0.5, 2.0, 5.0, 10]


'''p1_hist, p2_hist, _, _, _ = play_game(rounds=500, epsilon=0.01, delta = 2)

plt.plot(p1_hist, label="P1 actions")
plt.plot(p2_hist, label="P2 actions")
plt.xlabel("Round")
plt.ylabel("Action (1=C, 0=D)")
plt.legend()
plt.title("Actions over time")
plt.show()'''


'''(
    p1_hist,
    p2_hist,
    _,
    _,
    prob_dist
) = play_game(rounds=10000, epsilon=0.01, delta=2)

t_eq = equilibrium_time(prob_dist, eps=0.0003, min_t=1000, persistence=100) # from eyesight 0.0004 seems the best error value
print(f"Learning equilibrium reached at round t = {t_eq}")

plt.figure(figsize=(12,6))
for i, js in enumerate(joint_strats):
    plt.plot(prob_dist[:, i], label=f"{js[0]}, {js[1]}")

plt.xlabel("Round")
plt.ylabel("Probability")
plt.title("Joint strategy probability distribution over gameplay")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()'''


results = {}

for delta in deltas:
    (
        _,
        _,
        prob_dist
    ) = play_game(rounds=1500, epsilon=0.01, delta=delta)

    t_eq = equilibrium_time(
        prob_dist,
        eps=0.001,
        min_t=1000,
        persistence=100
    )

    results[delta] = {
        "prob_dist": prob_dist,
        "t_eq": t_eq
    }

    print(f"δ = {delta}: equilibrium at t = {t_eq}")


# --- colour choices ---
BG_COLOUR = "#02134A" 
FG_COLOUR = "#D5FAFF" 
FLURO_COLOURS = [
    "#00e5ff",  # cyan
    "#ff00ff",  # magenta
    "#39ff14",  # neon green
    "#ffea00",  # neon yellow
    "#ff6f00",  # neon orange
    "#ff1744"   # neon red
]

for delta, res in results.items():
    prob_dist = res["prob_dist"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- background ---
    fig.patch.set_facecolor(BG_COLOUR)
    ax.set_facecolor(BG_COLOUR)

    # --- plot lines ---
    for i, js in enumerate(joint_strats):
        ax.plot(
            prob_dist[:, i],
            label=f"{js[0]}, {js[1]}",
            color=FLURO_COLOURS[i % len(FLURO_COLOURS)],
            linewidth=2.2
        )

    # --- labels & title ---
    ax.set_xlabel("Round", color=FG_COLOUR, fontsize=12)
    ax.set_ylabel("Probability", color=FG_COLOUR, fontsize=12)
    ax.set_title(f"Joint strategy distribution (δ = {delta})",
                 color=FG_COLOUR, fontsize=14, pad=10)

    # --- ticks ---
    ax.tick_params(colors=FG_COLOUR)

    # --- spines ---
    for spine in ax.spines.values():
        spine.set_color(FG_COLOUR)

    # --- legend ---
    legend = ax.legend(ncol=3, fontsize=8, frameon=True)
    legend.get_frame().set_facecolor(BG_COLOUR)
    legend.get_frame().set_edgecolor(FG_COLOUR)
    for text in legend.get_texts():
        text.set_color(FG_COLOUR)

    plt.tight_layout()
    plt.show()








### TO DO
# Check random range for checking p_switch
# Set a random seed



