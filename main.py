import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime
from Env import Env
from Model import train, save, load
from Analyze import test

envTrain = Env("Data/select_train_data_30m_2.csv")
envTest = Env("Data/select_test_data_30m_2.csv")

NB_NEURONS = 10
NB_EPISODES = 1000
NB_STEPS = 10
BATCH_SIZE = 100

model_name = f"{NB_NEURONS}nn_{NB_EPISODES}ep_{NB_STEPS}s_{BATCH_SIZE}b"

print("Training...")
DQN = train(
    envTrain,
    n_neurons=NB_NEURONS,
    nb_episodes=NB_EPISODES,
    nb_steps=NB_STEPS,
    batch_size=BATCH_SIZE,
    model_name=model_name,
    # save_step=50,
    # recup_model=True,
)
print("Done")

test(envTest, DQN_model=DQN)
