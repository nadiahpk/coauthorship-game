import numpy as np
import itertools
import random
import matplotlib.pyplot as plt

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

# Starting Strategies of both players
p1_strat = strategies["ALLC"]
p2_strat = strategies["ALLC"]


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




def checkstrat ():
    ### Compares strategies between two current players

    global p1_strat, p2_strat, strategies
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
    payoff_r = sum(v_r[i] * U1[states[i]] if learner == 1 else v_r[i] * U2[states[i]] for i in range(4))
    payoff_l = sum(v_l[i] * U1[states[i]] if learner == 1 else v_l[i] * U2[states[i]] for i in range(4))

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
        p1c = p1[i] # prob p1 cooperates given previous round
        p2c = p2[i] # prob p2 cooperates given previous round

        M[i, state_index[(1,1)]] = p1c * p2c
        M[i, state_index[(1,0)]] = p1c * (1 - p2c)
        M[i, state_index[(0,1)]] = (1 - p1c) * p2c
        M[i, state_index[(0,0)]] = (1 - p1c) * (1 - p2c)

    return M



def play_game(rounds, epsilon, delta):
    global p1_strat, p2_strat
    a1, a2 = 1, 1  # start CC (try 16 combos or try random a couple of times)
    history_states = []
    # To record action history
    p1_history = []
    p2_history = []
    # To record strategy history
    p1_strat_hist = []
    p2_strat_hist = []

    # Probability Distribution for joint strategies
    joint_counts = np.zeros(len(joint_strats))
    prob_dist_stratpairs = []

    for t in range(rounds):
        

        # choose actions
        i = state_index[(a1, a2)]
        a1_goal = p1_strat[i]
        a2_goal = p2_strat[i]


        # implementation error? // NEED TO CHECK
        a1 = 1 - a1_goal if np.random.rand() < epsilon else a1_goal
        a2 = 1 - a2_goal if np.random.rand() < epsilon else a2_goal

        p1_history.append(a1)
        p2_history.append(a2)

        # record current strategies
        p1_strat_hist.append(p1_strat.copy())
        p2_strat_hist.append(p2_strat.copy())

        # Record joint-strats and calculate prob_dist for this round
        current_joint_strat = (
            strat_name_from_vec(p1_strat),
            strat_name_from_vec(p2_strat)
        )
        joint_counts[joint_index[current_joint_strat]] += 1

        prob_dist_stratpairs.append(
            joint_counts / (t + 1)
        )

        # Introspect at the end of each round
        p1_strat, p2_strat = checkstrat()

    return (p1_history,p2_history,p1_strat_hist,p2_strat_hist,np.array(prob_dist_stratpairs))


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
        p1_hist,
        p2_hist,
        _,
        _,
        prob_dist
    ) = play_game(rounds=10000, epsilon=0.01, delta=delta)

    t_eq = equilibrium_time(
        prob_dist,
        eps=0.0003,
        min_t=1000,
        persistence=100
    )

    results[delta] = {
        "prob_dist": prob_dist,
        "t_eq": t_eq
    }

    print(f"δ = {delta}: equilibrium at t = {t_eq}")

for delta, res in results.items():
    prob_dist = res["prob_dist"]

    plt.figure(figsize=(12, 6))
    for i, js in enumerate(joint_strats):
        plt.plot(prob_dist[:, i], label=f"{js[0]}, {js[1]}")

    plt.xlabel("Round")
    plt.ylabel("Probability")
    plt.title(f"Joint strategy distribution (δ = {delta})")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()







### TO DO
# Plot change in strats over games (strat distribution) - 3 graphs (epsilon, delta constant; testing epsilon, testing deltas)
# find equilibrium time (no. of games to stabilise) slice that time and check for introspection strength and probabilities
# Test for different values of epsilon in implementation error
# Try running the game with different start values (either randomly or 16 combinations)
# Check random range for checking p_switch



