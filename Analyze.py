import numpy as np
import matplotlib.pyplot as plt
import copy

from Env import Env, ACTIONS
from Model import predict

"""
Strategies supported :
    - DQN
    - Random
    - Nothing
    - RandomBattery    # random charge/discharge
    - SmartBattery
"""
STRATEGIES = ["DQN", "Random", "Nothing", "RandomBattery", "SmartBattery"]


def strategyAction(strategy, state, DQN_model=None):
    if strategy == "DQN":
        if DQN_model is None:
            print("No DQN model given in the function strategyAction")
            return ACTIONS[0]

        # Deterministic
        q_value = [predict(DQN_model, state, a) for a in ACTIONS]
        return ACTIONS[np.argmax(q_value)]

        # Stochastic
        # action_probs = policy(DQN_model, env.currentState)
        # return np.random.choice(ACTIONS, p=action_probs)

    if strategy == "Random":
        return np.random.choice(ACTIONS)

    if strategy == "Nothing":
        return ACTIONS[-1]

    if strategy == "RandomBattery":
        return np.random.choice(ACTIONS[0:2])

    if strategy == "SmartBattery":
        if state.panelProd > state.consumption:
            return ACTIONS[0]
        else:
            return ACTIONS[1]


def test(env: Env, nb_step=3000, DQN_model=None):
    env.initState(maxNbStep=nb_step)
    initState = copy.deepcopy(env.currentState)

    conso, prod, price = [], [], []

    actions_qvalue = {}
    for a in ACTIONS:
        actions_qvalue[a] = []

    for i in range(nb_step):
        env.act(ACTIONS[0])

        conso.append(env.currentState.consumption)
        prod.append(env.currentState.panelProd)
        price.append(env.currentState.price)

    actions, cost = {}, {}
    battery, charge, discharge, generate, trade = {}, {}, {}, {}, {}

    strategies_list = STRATEGIES[:]

    if DQN_model is None:
        strategies_list.remove("DQN")

    for strategy in strategies_list:
        actions[strategy], cost[strategy] = [], []
        (
            battery[strategy],
            charge[strategy],
            discharge[strategy],
            generate[strategy],
            trade[strategy],
        ) = ([], [], [], [], [])

        env.currentState = copy.deepcopy(initState)
        for i in range(nb_step):
            if strategy == "DQN":
                q_value = [predict(DQN_model, env.currentState, a) for a in ACTIONS]
                action = ACTIONS[np.argmax(q_value)]
                for q, a in zip(q_value, ACTIONS):
                    actions_qvalue[a].append(float(q))
            else:
                action = strategyAction(strategy, env.currentState)
            reward, next_state = env.act(action)

            cost[strategy].append(-reward)
            actions[strategy].append(action)
            battery[strategy].append(env.currentState.battery)

            charge[strategy].append(env.currentState.charge)
            discharge[strategy].append(env.currentState.discharge)
            generate[strategy].append(env.currentState.generate)
            trade[strategy].append(env.currentState.trade)

    fig, axs = plt.subplots(len(strategies_list))
    for i, s in enumerate(strategies_list):
        axs[i].plot(trade[s])
        # axs[i].plot(generate[s])
        axs[i].plot(battery[s])
        # axs[i].legend(["Trade", "Generator", "Battery"])
        axs[i].legend(["Trade", "Battery"])
        axs[i].title.set_text(s)
    plt.figure(1)

    fig, axs = plt.subplots(len(strategies_list))

    for i, s in enumerate(strategies_list):
        axs[i].plot(actions[s])
        axs[i].legend(["Actions"])
        axs[i].title.set_text(s)
    plt.figure(2)

    fig, axs = plt.subplots(2)
    axs[0].plot(conso)
    axs[0].plot(prod)
    axs[1].plot(price)
    axs[0].legend(["Consumption", "Production"])
    axs[1].title.set_text("Price")
    plt.figure(3)

    fig, ax = plt.subplots()
    for s in strategies_list:
        ax.plot(np.cumsum(cost[s]))

    ax.legend(strategies_list)
    ax.title.set_text("Cost")
    plt.figure(4)

    fig, ax = plt.subplots()
    for a in ACTIONS:
        ax.plot(actions_qvalue[a])
    ax.legend(ACTIONS)
    ax.title.set_text("Q-value")
    plt.figure(5)

    plt.show()
