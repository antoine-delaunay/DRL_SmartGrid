import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import gym
import datetime
from statistics import mean
from gym import wrappers

ACTIONS = np.array(["charge", "discharge", "trade"])
N_DISCRETE_ACTIONS = len(ACTIONS)


class State:
    def __init__(self):
        self.battery = 0.0
        self.panelProd = 0.0
        self.consumption = 0.0

    def toArray(self):
        return np.array([self.panelProd, self.consumption, self.battery])


class Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Env, self).__init__()

        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        )

        self.currentState = State()
        self.currentState = 0

        self.charge = 0.0
        self.discharge = 0.0
        self.generate = 0.0
        self.trade = 0.0

        self.batteryCapacity = 1  # 60000.0 / self.panelProdMax

    def reset(self):
        self.currentState = State()

        theta = 2 * np.pi * np.random.rand()
        self.currentState.consumption = np.cos(theta)
        self.currentState.panelProd = np.sin(theta)

        return self.currentState.toArray()

    def step(self, action):
        diffProd = self.currentState.panelProd - self.currentState.consumption
        cost = 0.0
        self.charge = 0.0
        self.discharge = 0.0
        self.trade = 0.0

        if action == 0:  # charge:
            if diffProd > 0:
                self.charge = min(diffProd, (self.batteryCapacity - self.currentState.battery))
                self.currentState.battery += self.charge
                diffProd -= self.charge
                # cost += self.currentState.charge * self.chargingCost

        elif action == 1:  # discharge
            if diffProd < 0:
                self.discharge = max(diffProd, -self.currentState.battery)
                self.currentState.battery += self.discharge
                diffProd -= self.discharge

        self.trade = -diffProd

        if diffProd < -1e-3:
            cost = 1.0

        angle = np.pi / 4
        O = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        self.currentState.consumption, self.currentState.panelProd = O @ np.array(
            [self.currentState.consumption, self.currentState.panelProd]
        )

        done = False

        return self.currentState.toArray(), -cost, done, {}

    def render(self, mode="human", close=False):
        print("ok")


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(i, activation="tanh", kernel_initializer="RandomNormal")
            )
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation="linear", kernel_initializer="RandomNormal"
        )

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(
        self,
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        lr,
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {"s": [], "a": [], "r": [], "s2": [], "done": []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype("float32")))

    def train(self, TargetNet):
        if len(self.experience["s"]) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience["s"]), size=self.batch_size)
        states = np.asarray([self.experience["s"][i] for i in ids])
        actions = np.asarray([self.experience["a"][i] for i in ids])
        rewards = np.asarray([self.experience["r"][i] for i in ids])
        states_next = np.asarray([self.experience["s2"][i] for i in ids])
        dones = np.asarray([self.experience["done"][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1
            )
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience["s"]) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def epoch(env, TrainNet, TargetNet, epsilon, copy_step, nb_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    for _ in range(nb_step):
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, _, _ = env.step(action)
        rewards += reward

        exp = {"s": prev_observations, "a": action, "r": reward, "s2": observations, "done": done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)


if __name__ == "__main__":
    env = Env()
    # env = gym.make("CartPole-v0")
    gamma = 0.99
    copy_step = 25
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/dqn/" + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        lr,
    )
    TargetNet = DQN(
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        lr,
    )

    N = 10000
    nb_step = 100
    # N = 50000
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = epoch(env, TrainNet, TargetNet, epsilon, copy_step, nb_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100) : (n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar("episode reward", total_reward, step=n)
            tf.summary.scalar("running avg reward(100)", avg_rewards, step=n)
            tf.summary.scalar("average loss", losses, step=n)
        if n % 100 == 0:
            print(
                f"episode: {n} eps: {epsilon} avg reward (last 100): {avg_rewards} episode loss: {losses}"
            )
    print("avg reward for last 100 episodes:", avg_rewards)
    env.close()
