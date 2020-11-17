from Env import *
from Model import *

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


def test(env: Env, DQN_model):
    env.initState()
    initState = copy.deepcopy(env.currentState)

    conso, prod, price = {}, {}, {}
    actions, cost = {}, {}
    battery, charge, discharge, generate, trade = {}, {}, {}, {}, {}

    for strategy in STRATEGIES:
        conso[strategy], prod[strategy], price[strategy] = [], [], []
        actions[strategy], cost[strategy] = [], []
        (
            battery[strategy],
            charge[strategy],
            discharge[strategy],
            generate[strategy],
            trade[strategy],
        ) = ([], [], [], [], [])

        env.currentState = copy.deepcopy(initState)

        for i in range(3000):
            if strategy == "DQN":
                action = strategyAction(strategy, env.currentState, DQN_model)
            else:
                action = strategyAction(strategy, env.currentState)
            reward, next_state = env.act(action)

            conso[strategy].append(env.currentState.consumption),
            prod[strategy].append(env.currentState.panelProd),
            price[strategy].append(env.currentState.price)

            cost[strategy].append(-reward)
            actions[strategy].append(action)
            battery[strategy].append(env.currentState.battery)

            charge[strategy].append(env.currentState.charge)
            discharge[strategy].append(env.currentState.discharge)
            generate[strategy].append(env.currentState.generate)
            trade[strategy].append(env.currentState.trade)

    fig, axs = plt.subplots(len(STRATEGIES))
    for i, s in enumerate(STRATEGIES):
        axs[i].plot(trade[s])
        axs[i].plot(generate[s])
        axs[i].plot(battery[s])
        axs[i].legend(["Trade", "Generator", "Battery"])
        axs[i].title.set_text(s)
    plt.figure(1)

    fig, axs = plt.subplots(len(STRATEGIES))
    for i, s in enumerate(STRATEGIES):
        axs[i].plot(actions[s])
        axs[i].legend(["Actions"])
        axs[i].title.set_text(s)
    plt.figure(2)

    fig, axs = plt.subplots(len(STRATEGIES))
    for i, s in enumerate(STRATEGIES):
        axs[i].plot(conso[s])
        axs[i].plot(prod[s])
        axs[i].plot(battery[s])
        axs[i].plot(price[s])
        axs[i].legend(["Consumption", "Production", "Battery"])
        axs[i].title.set_text(s)
    plt.figure(3)

    fig, ax = plt.subplots()
    for s in STRATEGIES:
        ax.plot(np.cumsum(cost[s]))
    ax.legend(STRATEGIES)
    ax.title.set_text("Cost")
    plt.figure(4)

    plt.show()
