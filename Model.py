import numpy as np
import random
import datetime
import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.utils import plot_model

from Env import Env, ACTIONS, State

NB_ACTION = len(ACTIONS)
DIM_STATE = len(State().toArray())
EPS = 0.8
GAMMA = 0.8


def DQN(n_neurons, input_size):
    model = tf.keras.Sequential(name="DQN")
    model.add(layers.Dense(n_neurons, input_shape=(input_size,), activation="sigmoid"))
    model.add(layers.Dense(n_neurons, activation="sigmoid"))
    model.add(layers.Dense(units=1))
    return model


def save(model, name):
    model.save(name)


def load(name):
    return tf.keras.models.load_model(name)


def predict_list(model, state_action_list):
    def input_one_action(state, action):
        input_action = np.zeros(NB_ACTION)
        input_action[ACTIONS == action] = 1.0
        return np.concatenate((input_action, state.toArray()))

    input_model = []
    for state, action in state_action_list:
        if isinstance(action, (list, np.ndarray)):
            for a in action:
                input_model.append(input_one_action(state, a))
        else:
            input_model.append(input_one_action(state, action))

    return model(np.array(input_model))


def predict(model, state, action):
    return predict_list(model, [(state, action)])


def policy(model, state):
    q_value = predict(model, state, ACTIONS)
    prob = np.ones(NB_ACTION) * EPS / NB_ACTION
    prob[np.argmax(q_value)] += 1.0 - EPS
    return prob


def loss(model, transitions_batch):
    state_action_list_y = [(next_state, ACTIONS) for _, _, _, next_state in transitions_batch]
    state_action_list_q = [(state, action) for state, action, _, _ in transitions_batch]

    batch_comp = predict_list(model, state_action_list_q + state_action_list_y)
    q = tf.reshape(batch_comp[: len(state_action_list_q)], [-1])
    y_precomp = batch_comp[len(state_action_list_q) :]

    y = [
        reward + GAMMA * np.max(y_precomp[NB_ACTION * i : NB_ACTION * (i + 1)])
        for i, (_, _, reward, _) in enumerate(transitions_batch)
    ]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")
    return tf.square(q - tf.stop_gradient(y))


def loss_double(model_A, model_B, transitions_batch):
    state_action_list_y = [(next_state, ACTIONS) for _, _, _, next_state in transitions_batch]
    state_action_list_q = [(state, action) for state, action, _, _ in transitions_batch]

    q = tf.reshape(predict_list(model_A, state_action_list_q), [-1])

    y_q_A = predict_list(model_A, state_action_list_y)
    state_action_list_model_B = [
        (next_state, ACTIONS[np.argmax(y_q_A[NB_ACTION * i : NB_ACTION * (i + 1)])])
        for i, (_, _, _, next_state) in enumerate(transitions_batch)
    ]
    y_q_B_batch = predict_list(model_B, state_action_list_model_B)

    y = [reward + GAMMA * q_B for q_B, (_, _, reward, _) in zip(y_q_B_batch, transitions_batch)]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")
    return tf.square(q - tf.stop_gradient(y))


def train_step(model, transitions_batch, optimizer):
    with tf.GradientTape() as disc_tape:
        disc_loss = loss(model, transitions_batch)

    gradients = disc_tape.gradient(disc_loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return disc_loss


def train_step_double(model_A, model_B, transitions_batch, optimizer):
    with tf.GradientTape() as disc_tape:
        disc_loss = loss_double(model_A, model_B, transitions_batch)

    gradients = disc_tape.gradient(disc_loss, model_A.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

    optimizer.apply_gradients(zip(gradients, model_A.trainable_variables))
    return disc_loss


def train(
    env: Env,
    n_neurons,
    nb_episodes=50,
    nb_steps=50,
    batch_size=100,
    model_name=None,
    save_step=None,
    recup_model=False,
    algo="simple",
):
    alpha = 0.7

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"logs/{model_name}_{current_time}/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    train_reward = tf.keras.metrics.Mean("train_reward", dtype=tf.float32)
    train_qvalues = {}
    for a in ACTIONS:
        train_qvalues[a] = tf.keras.metrics.Mean("train_" + a, dtype=tf.float32)

    input_size = DIM_STATE + NB_ACTION
    DQN_model = []
    if model_name and recup_model:
        DQN_model.append(load(f"models/{model_name}"))
        print("Model loaded")
        # we have to check if the model loaded has the same input size than the one expected here
    else:
        DQN_model.append(DQN(n_neurons=n_neurons, input_size=input_size))

    if algo == "double":
        DQN_model.append(DQN(n_neurons=n_neurons, input_size=input_size))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    replay_memory = []
    replay_memory_init_size = 10 * batch_size

    env.initState(maxNbStep=replay_memory_init_size)
    for i in range(replay_memory_init_size):
        action_probs = policy(DQN_model[0], env.currentState)
        action = np.random.choice(ACTIONS, p=action_probs)
        reward, next_state = env.act(action)
        replay_memory.append((env.currentState, action, reward, next_state))

    for i_episode in range(nb_episodes):
        env.initState(maxNbStep=nb_steps)
        if i_episode % 10 == 0:
            print(i_episode)

        if algo == "double":
            if np.random.rand() > 0.5:
                DQN_model[0], DQN_model[1] = DQN_model[1], DQN_model[0]

        # total_reward = 0
        for step in range(nb_steps):
            action_probs = policy(DQN_model[0], env.currentState)
            action = np.random.choice(ACTIONS, p=action_probs)
            reward, next_state = env.act(action)

            # if step == 0:
            #     total_reward = reward
            # else:
            #     total_reward = (1 - alpha) * total_reward + alpha * reward

            replay_memory.pop(0)
            replay_memory.append((env.currentState, action, reward, next_state))

            samples = random.sample(replay_memory, batch_size)
            if algo == "double":
                loss_step = train_step_double(DQN_model[0], DQN_model[1], samples, optimizer)
            else:
                loss_step = train_step(DQN_model[0], samples, optimizer)
            train_loss(loss_step)

        # Test phase : compute reward
        env.initState(maxNbStep=nb_steps)
        for step in range(nb_steps):
            q_value = predict(DQN_model[0], env.currentState, ACTIONS)
            action = ACTIONS[np.argmax(q_value)]
            reward, _ = env.act(action)
            train_reward(reward)
            for q, a in zip(q_value, ACTIONS):
                train_qvalues[a](q)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=i_episode)
            tf.summary.scalar("reward", train_reward.result(), step=i_episode)
            for a in ACTIONS:
                tf.summary.scalar("Q value : " + a, train_qvalues[a].result(), step=i_episode)

        train_loss.reset_states()
        train_reward.reset_states()
        for a in ACTIONS:
            train_qvalues[a].reset_states()

        if model_name and save_step and (i_episode + 1) % save_step == 0:
            save(DQN_model[0], f"models/{model_name}")
            print("Model saved")

    if model_name and save_step:
        save(DQN_model[0], f"models/{model_name}")
        print("Model saved")
    return DQN_model[0]
