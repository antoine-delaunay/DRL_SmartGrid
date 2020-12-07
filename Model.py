import numpy as np
import random
import datetime
import copy
import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.utils import plot_model

from Env import Env, ACTIONS, State

NB_ACTION = len(ACTIONS)
DIM_STATE = len(State().toArray())
GAMMA = 0.5


def build_NN(input_size, output_size, hidden_layers=[]):
    """ 
    Builds a sequential neural network.
    The shapes are defined by the parameters.
    The activation function used is the sigmoid.

    Parameters: 
    input_size (int)

    output_size (int)

    hidden_layers (list)

    Returns: 
    model: a neural network built with TensorFlow 2

    """
    model = tf.keras.Sequential()

    hidden_layers = hidden_layers[:]
    hidden_layers.append(output_size)
    input_layer = input_size

    while len(hidden_layers) > 1:
        output_layer = hidden_layers.pop(0)
        model.add(layers.Dense(output_layer, input_shape=(input_layer,), activation="sigmoid"))
        input_layer = output_layer

    output_layer = hidden_layers.pop(0)
    model.add(layers.Dense(output_layer))

    return model


def save(model, name):
    """ 
    Saves the model in a folder named name.

    Parameters: 
    model: a TensorFlow 2 model

    name (str): name of the model

    """
    model.save(name)


def load(name):
    """ 
    Loads a TensorFlow 2 model from a folder named name

    Parameters: 
    name (str): name of the model

    Returns: 
    model: a neural network built with TensorFlow 2

    """
    return tf.keras.models.load_model(name)


def predict_list(model, state_action_list):
    """ 
    Returns the Q-values predicted by the model and associated to the pairs (state, action)
    in the list state_action_list.

    Parameters: 
    model: a neural network built with TensorFlow 2

    state_action_list (list): list of pairs (state, action) 

    Returns: 
    Q_values (tf.Tensor): Q_values predicted by the model

    """

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
    """ 
    Returns the Q-value predicted by the model for the pair (state, action).

    Parameters: 
    model: a neural network built with TensorFlow 2

    state (State)

    action (str) 

    Returns: 
    Q_value (tf.Tensor): Q_value predicted by the model

    """
    return predict_list(model, [(state, action)])


def eps_greedy_policy(model, state, epsilon):
    """ 
    Returns the probability array associated to the epsilon greedy policy.

    Parameters: 
    model: a neural network built with TensorFlow 2

    state (State)

    espilon (float)

    Returns: 
    prob (np.array): probability array

    """
    q_value = predict(model, state, ACTIONS)
    prob = np.ones(NB_ACTION) * epsilon / NB_ACTION
    prob[np.argmax(q_value)] += 1.0 - epsilon
    return prob


def loss(model, target_model, transitions_batch):
    """ 
    Computes the loss of the model according to the predictions of the model and
    the target model on the transitions_batch. The loss computed is the one associated to 
    the DQN algorithm with a target estimator.

    Parameters: 
    model: a neural network built with TensorFlow 2

    target_model: a neural network built with TensorFlow 2

    transitions_batch (list): list of tuples (state, action, reward, next_state) 

    Returns: 
    loss (tf.Tensor)

    """
    state_action_list_y = [(next_state, ACTIONS) for _, _, _, next_state, _ in transitions_batch]
    state_action_list_q = [(state, action) for state, action, _, _, _ in transitions_batch]

    q = tf.reshape(predict_list(model, state_action_list_q), [-1])

    batch_target = predict_list(target_model, state_action_list_y)
    y = [
        reward + GAMMA ** k * np.max(batch_target[NB_ACTION * i : NB_ACTION * (i + 1)])
        for i, (_, _, reward, _, k) in enumerate(transitions_batch)
    ]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")
    # return tf.square(q - tf.stop_gradient(y))


def loss_double(model_A, model_B, transitions_batch):
    """ 
    Computes the loss of the model_A according to the predictions of the model_A and
    the model_B on the transitions_batch. The loss computed is the one associated to 
    the double DQN algorithm.

    Parameters: 
    model: a neural network built with TensorFlow 2

    target_model: a neural network built with TensorFlow 2

    transitions_batch (list): list of tuples (state, action, reward, next_state) 

    Returns: 
    loss (tf.Tensor)

    """
    state_action_list_y = [(next_state, ACTIONS) for _, _, _, next_state, _ in transitions_batch]
    state_action_list_q = [(state, action) for state, action, _, _, _ in transitions_batch]

    q = tf.reshape(predict_list(model_A, state_action_list_q), [-1])

    y_q_A = predict_list(model_A, state_action_list_y)

    state_action_list_model_B = [
        (next_state, ACTIONS[np.argmax(y_q_A[NB_ACTION * i : NB_ACTION * (i + 1)])])
        for i, (_, _, _, next_state, _) in enumerate(transitions_batch)
    ]
    y_q_B_batch = predict_list(model_B, state_action_list_model_B)

    y = [
        reward + GAMMA ** k * q_B
        for q_B, (_, _, reward, _, k) in zip(y_q_B_batch, transitions_batch)
    ]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    return tf.reduce_mean(tf.square(q - tf.stop_gradient(y)), name="loss_mse_train")
    # return tf.square(q - tf.stop_gradient(y))


def train_step(model, target_model, transitions_batch, optimizer):
    """ 
    Trains the model on the transitions_batch using the target_model and the parameters defined
    in the optimizer. This is the training associated to the DQN algorithm with a target model.

    Parameters: 
    model: a neural network built with TensorFlow 2

    target_model: a neural network built with TensorFlow 2

    transitions_batch (list): list of tuples (state, action, reward, next_state) 

    optimizer: an optimizer instantiated with TensorFlow 2

    Returns: 
    loss (tf.Tensor)

    """
    with tf.GradientTape() as disc_tape:
        disc_loss = loss(model, target_model, transitions_batch)

    gradients = disc_tape.gradient(disc_loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return disc_loss


def train_step_double(model_A, model_B, transitions_batch, optimizer):
    """ 
    Trains the model_A on the transitions_batch using the model_B and the parameters defined
    in the optimizer. This is the training associated to the Double DQN algorithm.

    Parameters: 
    model_A: a neural network built with TensorFlow 2

    model_B: a neural network built with TensorFlow 2

    transitions_batch (list): list of tuples (state, action, reward, next_state) 

    optimizer: an optimizer instantiated with TensorFlow 2

    Returns: 
    loss (tf.Tensor)

    """
    with tf.GradientTape() as disc_tape:
        disc_loss = loss_double(model_A, model_B, transitions_batch)

    gradients = disc_tape.gradient(disc_loss, model_A.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

    optimizer.apply_gradients(zip(gradients, model_A.trainable_variables))
    return disc_loss


def train(
    env: Env,
    hidden_layers=[],
    nb_episodes=50,
    nb_steps=50,
    batch_size=100,
    model_name=None,
    save_episode=None,
    recup_model=False,
    algo="simple",
    y_method="monte_carlo",
    replay_memory_init_size=1000,
    replay_memory_size=100000,
    update_target_estimator_init=10,
    update_target_estimator_max=5000,
    update_target_estimator_epoch=50,
    epsilon_start=1.0,
    epsilon_min=0.4,
    epsilon_decay_steps=30000,
):
    """ 
    Builds and trains a model using a DQN algorithm.

    Parameters: 
    env: the environnement on which the model is trained

    hidden_layers: a list containing the number of neurons per hidden layer.
    The numbers of neurons at input and output are automatically deduced
    from the provided environment.

    nb_episodes: number of training episodes. An episode corresponds to a
    contiguous sequence of steps in the environment.
    
    nb_steps: number of steps per episode. A step corresponds to a call to
    the step function in the environment and to a training on a batch of the
    replay memory.

    batch_size: number of steps of the replay memory on which the network is
    trained for each new step generated.

    model_name: name of the model. This parameter characterizes the name of
    the saved model and the name that is displayed in TensorBoard.
    
    save_episode: the model is saved every save_episode episode. If this
    parameter is set to None then the template is not saved at all.
    
    recup_model: boolean defining whether or not to load a `model_name` model
    in the saves.
    
    algo: this parameter has two possible values: "single" or "double". It
    defines the algorithm to use. The first one corresponds to the classical
    DQN algorithm, while the second one corresponds to the double DQN algorithm.

    y_method: this parameter has two possible values: "monte_carlo" or "td". It
    defines the method to use to produce the y-value.
    
    replay_memory_init_size: initial size of the replay_memory.

    replay_memory_size: maximal/final size of the replay_memory.

    update_target_estimator_init: initial period of time during which the
    target estimator is not updated.
    
    update_target_estimator_max: maximum/final time period during which the
    target estimator is not updated.
    
    update_target_estimator_epoch: the update_target_estimator variable begins
    at update_target_estimator_init and ends at update_target_estimator_max.
    update_target_estimator_epoch determines the number of episodes to be done
    before increasing update_target_estimator.
    
    epsilon_start: initial value of the epsilon parameter in the DQN algorithm.
    
    epsilon_min: final value of the `epsilon` parameter in the DQN algorithm.

    epsilon_decay_steps: number of steps for epsilon to progress from
    epsilon_start to epsilon_min.

    Returns: 
    model: a neural network built with TensorFlow 2 and trained

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"logs/{model_name}_{current_time}/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    train_reward = tf.keras.metrics.Mean("train_reward", dtype=tf.float32)
    train_qvalues = {}
    for a in ACTIONS:
        train_qvalues[a] = tf.keras.metrics.Mean("train_" + a, dtype=tf.float32)

    input_size = DIM_STATE + NB_ACTION
    DQN_model = {}
    if model_name and recup_model:
        DQN_model["Q_estimator"] = load(f"models/{model_name}")
        print("Model loaded")
        # we have to check if the model loaded has the same input size than the one expected here
    else:
        DQN_model["Q_estimator"] = build_NN(
            input_size=input_size, output_size=1, hidden_layers=hidden_layers
        )

    if algo == "simple":
        DQN_model["target_estimator"] = build_NN(
            input_size=input_size, output_size=1, hidden_layers=hidden_layers
        )
    if algo == "double":
        DQN_model["Q_estimator_bis"] = build_NN(
            input_size=input_size, output_size=1, hidden_layers=hidden_layers
        )

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    epsilon = epsilon_start
    d_epsilon = (epsilon_start - epsilon_min) / float(epsilon_decay_steps)
    replay_memory = []
    # env.reset()
    # for i in range(replay_memory_init_size):
    #     next_state = None
    #     while next_state is None:
    #         action_probs = eps_greedy_policy(DQN_model["Q_estimator"], env.currentState, epsilon)
    #         action = np.random.choice(ACTIONS, p=action_probs)
    #         reward, next_state, _ = env.step(action)
    #         if next_state is not None:
    #             break
    #         env.reset()

    #     replay_memory.append((env.currentState, action, reward, next_state))

    update_target_estimator = update_target_estimator_init
    total_step = 0
    for i_episode in range(nb_episodes):
        env.reset(nb_step=nb_steps)
        print(f"epoch {i_episode}\teps {epsilon}\ttarget_update {update_target_estimator}")

        # Train phase
        if (i_episode + 1) % update_target_estimator_epoch == 0:
            update_target_estimator = min(5 * update_target_estimator, update_target_estimator_max)

        if algo == "double":
            if np.random.rand() > 0.5:
                DQN_model["Q_estimator"], DQN_model["Q_estimator_bis"] = (
                    DQN_model["Q_estimator_bis"],
                    DQN_model["Q_estimator"],
                )

        if y_method == "monte_carlo":
            state_list = []
            action_list = []
            reward_list = []
            last_state = None

        for step in range(nb_steps):
            if algo == "simple":
                if total_step % update_target_estimator == 0:
                    DQN_model["target_estimator"].set_weights(
                        DQN_model["Q_estimator"].get_weights()
                    )
                    print("Target estimator updated")

            action_probs = eps_greedy_policy(
                DQN_model["Q_estimator"], env.currentState, epsilon * np.random.rand()
            )
            action = np.random.choice(ACTIONS, p=action_probs)
            reward, next_state, _ = env.step(action)

            if y_method == "td":
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)
                replay_memory.append((env.currentState, action, reward, next_state, 1))

            if y_method == "monte_carlo":
                state_list.append(copy.deepcopy(env.currentState))
                action_list.append(action)
                reward_list.append(reward)
                last_state = copy.deepcopy(next_state)

            if len(replay_memory) > 0:
                samples = random.sample(replay_memory, min(batch_size, len(replay_memory)))
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

        if y_method == "monte_carlo":
            if len(replay_memory) == replay_memory_size:
                replay_memory = replay_memory[nb_steps:]

            gamma_power = [GAMMA ** i for i in range(nb_steps)]
            for i, (state, action) in enumerate(zip(state_list, action_list)):
                gamma_power_i = gamma_power[:-i] if i > 0 else gamma_power
                replay_memory.append(
                    (
                        state,
                        action,
                        np.sum(np.multiply(reward_list[i:], gamma_power_i)),
                        last_state,
                        nb_steps - i,
                    )
                )

        # Test phase : compute reward
        env.reset(nb_step=nb_steps)
        for step in range(nb_steps):
            q_value = predict(DQN_model["Q_estimator"], env.currentState, ACTIONS)
            action = ACTIONS[np.argmax(q_value)]
            reward, _, _ = env.step(action)
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

        if model_name and save_episode and (i_episode + 1) % save_episode == 0:
            save(DQN_model["Q_estimator"], f"models/{model_name}")
            print("Model saved")

    if model_name and save_episode:
        save(DQN_model["Q_estimator"], f"models/{model_name}")
        print("Model saved")
    return DQN_model["Q_estimator"]
