import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime
from Env import Env
from Model2 import train, save, load

# from Model import train, save, load
from Analyze import test

envTrain = Env("Data/select_train_data_30m_2.csv")
envTest = Env("Data/select_test_data_30m_2.csv")

ALGO = "simple"
NB_NEURONS = 200
NB_EPISODES = 1000
NB_STEPS = 32
BATCH_SIZE = 32

model_name = f"{ALGO}_{NB_NEURONS}nn_{NB_EPISODES}ep_{NB_STEPS}s_{BATCH_SIZE}b"

print("Training...")
DQN = train(
    envTrain,
    n_neurons=NB_NEURONS,
    nb_episodes=NB_EPISODES,
    nb_steps=NB_STEPS,
    batch_size=BATCH_SIZE,
    model_name=model_name,
    algo=ALGO,
    # save_step=200,
    # recup_model=True,
)
print("Done")

test(envTest, DQN_model=DQN)
