import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime
from Env import Env
from Model import train, save, load
from Analyze import test

envTrain = Env("Data/select_train_data_30m.csv")
envTest = Env("Data/select_test_data_30m.csv")

n_neurons = 10

model_name = "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"/NN_{n_neurons}"

print("Training...")
DQN = train(
    envTrain,
    n_neurons=n_neurons,
    nb_episodes=5000,
    nb_steps=10,
    batch_size=100,
    model_name=model_name,
)
print("Done")

test(envTest, DQN_model=DQN)
