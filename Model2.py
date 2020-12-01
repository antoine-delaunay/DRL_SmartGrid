import numpy as np
import random
import datetime
import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.utils import plot_model

from Env import Env, ACTIONS, State

NB_ACTION = len(ACTIONS)
DIM_STATE = len(State().toArray())
GAMMA = 0.9


def DQN(n_neurons, input_size, output_size):
    model = tf.keras.Sequential(name="DQN")
    model.add(layers.Dense(n_neurons, input_shape=(input_size,), activation="sigmoid"))
    model.add(layers.Dense(n_neurons, activation="sigmoid"))
    model.add(layers.Dense(output_size))
    return model


def save(model, name):
    model.save(name)


def load(name):
    return tf.keras.models.load_model(name)


def predict_array(model, state_list):
    state_array = np.array([s.toArray() for s in state_list])
    return model(state_array)


def predict(model, state):
    return predict_array(model, [state])[0]


def eps_greedy_policy(model, state, epsilon):
    q_value = predict(model, state)
    prob = np.ones(NB_ACTION) * epsilon / NB_ACTION
    prob[np.argmax(q_value)] += 1.0 - epsilon
    return prob


def loss(model, target_model, transitions_batch):
    state_array_y = np.array([next_state for _, _, _, next_state in transitions_batch])
    state_array_q = np.array([state for state, _, _, _ in transitions_batch])
    action_array_q = np.array([action for _, action, _, _ in transitions_batch])

    q_actions = predict_array(model, state_array_q)
    q = tf.convert_to_tensor([q[ACTIONS == a] for a, q in zip(action_array_q, q_actions)])

    batch_target = predict_array(target_model, state_array_y)
    y = [
        reward + GAMMA * np.max(batch_target[i])
        for i, (_, _, reward, _) in enumerate(transitions_batch)
    ]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")
    # return tf.square(q - tf.stop_gradient(y))


def loss_double(model_A, model_B, transitions_batch):
    state_array_y = np.array([next_state for _, _, _, next_state in transitions_batch])
    state_array_q = np.array([state for state, _, _, _ in transitions_batch])
    action_array_q = np.array([action for _, action, _, _ in transitions_batch])

    q_actions = predict_array(model_A, state_array_q)
    q = tf.convert_to_tensor([q[ACTIONS == a] for a, q in zip(action_array_q, q_actions)])

    batch_A = predict_array(model_A, state_array_y)
    batch_B = predict_array(model_B, state_array_y)
    action_array_B = np.array(
        [ACTIONS[np.argmax(batch_A[i])] for i, _ in enumerate(transitions_batch)]
    )
    y_q_B_batch = tf.convert_to_tensor([b[ACTIONS == a] for a, b in zip(action_array_B, batch_B)])

    y = [reward + GAMMA * q_B for q_B, (_, _, reward, _) in zip(y_q_B_batch, transitions_batch)]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")
    return tf.square(q - tf.stop_gradient(y))


def train_step(model, target_model, transitions_batch, optimizer):
    with tf.GradientTape() as disc_tape:
        disc_loss = loss(model, target_model, transitions_batch)

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
    replay_memory_init_size=100,
    replay_memory_size=10000,
    update_target_estimator_every=32,
    epsilon_start=1.0,
    epsilon_min=0.1,
    epsilon_decay_steps=100000,
):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"logs/{model_name}_{current_time}/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    train_reward = tf.keras.metrics.Mean("train_reward", dtype=tf.float32)
    train_qvalues = {}
    for a in ACTIONS:
        train_qvalues[a] = tf.keras.metrics.Mean("train_" + a, dtype=tf.float32)

    input_size = DIM_STATE
    output_size = NB_ACTION
    DQN_model = {}
    if model_name and recup_model:
        DQN_model["Q_estimator"] = load(f"models/{model_name}")
        print("Model loaded")
        # we have to check if the model loaded has the same input size than the one expected here
    else:
        DQN_model["Q_estimator"] = DQN(
            n_neurons=n_neurons, input_size=input_size, output_size=output_size
        )

    if algo == "simple":
        DQN_model["target_estimator"] = DQN(
            n_neurons=n_neurons, input_size=input_size, output_size=output_size
        )
    if algo == "double":
        DQN_model["Q_estimator_bis"] = DQN(
            n_neurons=n_neurons, input_size=input_size, output_size=output_size
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    epsilon = epsilon_start
    d_epsilon = (epsilon_start - epsilon_min) / float(epsilon_decay_steps)
    replay_memory = []
    env.reset()
    for i in range(replay_memory_init_size):
        next_state = None
        while next_state is None:
            action_probs = eps_greedy_policy(DQN_model["Q_estimator"], env.currentState, epsilon)
            action = np.random.choice(ACTIONS, p=action_probs)
            reward, next_state = env.step(action)
            if next_state is not None:
                break
            env.reset()

        replay_memory.append((env.currentState, action, reward, next_state))

    total_step = 0
    for i_episode in range(nb_episodes):
        env.reset(nb_step=nb_steps)
        if i_episode % 10 == 0:
            print(i_episode)

        # Train phase
        if algo == "double":
            if np.random.rand() > 0.5:
                DQN_model["Q_estimator"], DQN_model["Q_estimator_bis"] = (
                    DQN_model["Q_estimator_bis"],
                    DQN_model["Q_estimator"],
                )

        for step in range(nb_steps):
            if algo == "simple":
                if total_step % update_target_estimator_every == 0:
                    DQN_model["target_estimator"].set_weights(
                        DQN_model["Q_estimator"].get_weights()
                    )

            action_probs = eps_greedy_policy(DQN_model["Q_estimator"], env.currentState, epsilon)
            action = np.random.choice(ACTIONS, p=action_probs)
            reward, next_state = env.step(action)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append((env.currentState, action, reward, next_state))

            samples = random.sample(replay_memory, batch_size)
            if algo == "simple":
                loss_step = train_step(
                    DQN_model["Q_estimator"], DQN_model["target_estimator"], samples, optimizer
                )
            if algo == "double":
                loss_step = train_step_double(
                    DQN_model["Q_estimator"], DQN_model["Q_estimator_bis"], samples, optimizer
                )
            train_loss(loss_step)

            total_step += 1

        epsilon = max(epsilon - d_epsilon, epsilon_min)

        # Test phase : compute reward
        env.reset(nb_step=nb_steps)
        for step in range(nb_steps):
            q_value = predict(DQN_model["Q_estimator"], env.currentState)
            action = ACTIONS[np.argmax(q_value)]
            reward, _ = env.step(action)
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
            save(DQN_model["Q_estimator"], f"models/{model_name}")
            print("Model saved")

    if model_name and save_step:
        save(DQN_model["Q_estimator"], f"models/{model_name}")
        print("Model saved")
    return DQN_model["Q_estimator"]
