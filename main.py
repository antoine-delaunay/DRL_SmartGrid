import datetime
from Env import Env
from Model import train, save, load
from Analyze import test

envTrain = Env("Data/select_train_data_30m.csv")
envTest = Env("Data/select_test_data_30m.csv")

model_name = "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/NN_10"

print("Training...")
DQN = train(envTrain, n_neurons=5, nb_episodes=100, nb_steps=10, batch_size=100,)
print("Done")

test(envTest, DQN_model=DQN)
