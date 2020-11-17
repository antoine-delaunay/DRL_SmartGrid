import pandas
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import random
import matplotlib.pyplot as plt
import numpy as np
import copy

ACTIONS = ["charge", "discharge", "generator", "discharge + generator", "nothing"]
NB_ACTION = len(ACTIONS)
EPS = 0.5
GAMMA = 1.0


class State:
    def __init__(self):
        self.battery = 0.0
        self.panelProd = 0.0
        self.consumption = 0.0
        self.price = 0.0
        self.daytime = 0.0
        self.row = 0

        self.charge = 0.0
        self.discharge = 0.0
        self.generate = 0.0
        self.trade = 0.0

    def toArray(self):
        return np.array(
            [self.battery, self.panelProd, self.consumption, self.price, 0]
        )  # self.daytime])


class Env:
    def __init__(self, dataFile: str):
        # load data (csv)
        df = pandas.read_csv(dataFile, sep=";", header=0)

        self.data = df.values

        # Prétraitement des données
        # TODO: transformer daytime en float
        self.panelProdMax = max(self.data[:, 5])
        self.consumptionMax = max(self.data[:, 4])

        self.data[:, 5] /= self.panelProdMax
        self.data[:, 4] /= self.consumptionMax
        self.data[:, 3] /= 1000.0

        # Capacity of the battery and the generator
        self.initState()
        self.batteryCapacity = 0.4  # 60000.0 / self.panelProdMax
        self.generatorCapacity = (
            0.4  # Energie produite par le générateur en 5min 20000.0 / (12 * self.panelProdMax)
        )

        # CO2 price/emissions
        self.co2Price = 25.0 * 0.001  # price per ton of CO2 (mean price from the european market)
        self.co2Generator = 8 * 0.001  # kg of CO2 generated per kWh from the diesel generator
        self.co2Market = (
            0.3204  # kg of CO2 generated per kWh from the national power market (danish)
        )

        # Operational costs
        self.chargingCost = 0.0
        self.dischargingCost = 0.0
        # self.solarCost = 0.0
        self.generatorCost = 0.4  # 0.314 à 0.528 $/kWh

        # Yields
        self.chargingYield = 1.0
        self.dischargingYield = 1.0

    def initState(self):
        self.currentState = State()
        self.currentState.row = np.random.randint(
            0, len(self.data)
        )  # Deuxième valeur à modifier en fonction du nombre de steps réalisés par épisode
        row = self.currentState.row
        self.currentState.daytime = self.data[row, 1]
        self.currentState.panelProd = self.data[row, 5]
        self.currentState.price = self.data[row, 3]
        self.currentState.consumption = self.data[row, 4]

    def act(self, action):
        self.diffProd = self.currentState.panelProd - self.currentState.consumption
        cost = 0.0
        self.currentState.charge = 0.0
        self.currentState.discharge = 0.0
        self.currentState.generate = 0.0
        self.currentState.trade = 0.0

        if action == "charge":
            if self.diffProd > 0:
                self.currentState.charge = min(
                    self.diffProd,
                    (self.batteryCapacity - self.currentState.battery) / self.chargingYield,
                )
                self.currentState.battery += self.currentState.charge * self.chargingYield
                self.diffProd -= self.currentState.charge
                cost += self.currentState.charge * self.chargingCost

        elif action == "discharge":
            if self.diffProd < 0:
                self.currentState.discharge = max(
                    self.diffProd / self.dischargingYield, -self.currentState.battery
                )
                self.currentState.battery += self.currentState.discharge
                self.diffProd -= self.currentState.discharge * self.dischargingYield
                cost += abs(self.currentState.discharge * self.dischargingCost)

        elif action == "generator":
            if self.diffProd < 0:
                self.currentState.generate = min(-self.diffProd, self.generatorCapacity)
                self.diffProd += self.currentState.generate
                cost += self.currentState.generate * self.generatorCost

        elif action == "discharge + generator":
            if self.diffProd < 0:
                self.currentState.discharge = max(
                    self.diffProd / self.dischargingYield, -self.currentState.battery
                )
                self.currentState.battery += self.currentState.discharge
                self.diffProd -= self.currentState.discharge * self.dischargingYield
                cost += abs(self.currentState.discharge * self.dischargingCost)

            if self.diffProd < 0:
                self.currentState.generate = min(-self.diffProd, self.generatorCapacity)
                self.diffProd += self.currentState.generate
                cost += self.currentState.generate * self.generatorCost

        self.currentState.trade = -self.diffProd

        cost -= self.diffProd * self.currentState.price

        # UPDATE SELF.PANELPROD, PRICE, CONSUMPTION, DAYTIME according to the dataset
        row = self.currentState.row + 1
        self.currentState.daytime = self.data[row, 1]
        self.currentState.panelProd = self.data[row, 5]
        self.currentState.price = self.data[row, 3]
        self.currentState.consumption = self.data[row, 4]
        self.currentState.row = row

        return -cost, self.currentState

    def getState(self):
        return self.currentState


def DQN(n_neurons, input_size):
    model = tf.keras.Sequential(name="DQN")
    model.add(
        layers.Dense(
            n_neurons,
            input_shape=(input_size,),
            bias_initializer="glorot_normal",
            kernel_initializer="glorot_normal",
        )
    )
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(
        layers.Dense(
            n_neurons, bias_initializer="glorot_normal", kernel_initializer="glorot_normal"
        )
    )
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(
        layers.Dense(units=1, bias_initializer="glorot_normal", kernel_initializer="glorot_normal")
    )
    return model


def predict(model, state, action):
    input_model = np.array([0.0] * NB_ACTION + list(state.toArray()))

    if action == "charge":
        input_model[0] = 1.0
    elif action == "discharge":
        input_model[1] = 1.0
    elif action == "generator":
        input_model[2] = 1.0
    elif action == "discharge + generator":
        input_model[3] = 1.0
    else:
        input_model[4] = 1.0

    return model(np.array([input_model]))


def policy(model, state):
    q_value = [predict(model, state, action) for action in ACTIONS]
    prob = np.ones(NB_ACTION) * EPS / NB_ACTION
    prob[np.argmax(q_value)] += 1.0 - EPS
    return prob


def loss(model, transitions_batch):
    y = []
    q = []
    for state, action, reward, next_state in transitions_batch:
        q_value = [predict(model, next_state, a) for a in ACTIONS]
        best_next_action = np.argmax(q_value)
        y.append(reward + GAMMA * q_value[best_next_action])
        q.append(predict(model, state, action))

    return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")


def train_step(model, transitions_batch, optimizer):
    with tf.GradientTape() as disc_tape:
        disc_loss = loss(model, transitions_batch)

    gradients = disc_tape.gradient(disc_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return disc_loss


def train(env: Env, nb_episodes=100, nb_steps=10, batch_size=10):
    DQN_model = DQN(n_neurons=10, input_size=10)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    replay_memory = []
    replay_memory_init_size = 100

    env.initState()
    for i in range(replay_memory_init_size):
        action_probs = policy(DQN_model, env.currentState)
        action = np.random.choice(ACTIONS, p=action_probs)
        reward, next_state = env.act(action)
        replay_memory.append((env.currentState, action, reward, next_state))

    loss_hist = []

    for i_episode in range(nb_episodes):
        env.initState()
        loss_episode = 0.0
        if i_episode % 10 == 0:
            print(i_episode)

        for step in range(nb_steps):
            action_probs = policy(DQN_model, env.currentState)
            action = np.random.choice(ACTIONS, p=action_probs)
            reward, next_state = env.act(action)

            replay_memory.pop(0)
            replay_memory.append((env.currentState, action, reward, next_state))

            samples = random.sample(replay_memory, batch_size)
            loss_episode += train_step(DQN_model, samples, optimizer)

        loss_hist.append(loss_episode)

    return (loss_hist, DQN_model)


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
    print(env.currentState.daytime)

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

        for i in range(300):
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


if __name__ == "__main__":
    envTrain = Env("select_train_data.csv")
    envTest = Env("select_test_data.csv")

    print("Training...")
    lossDQN, DQN = train(envTrain)
    print("Done")

    test(envTest, DQN)
