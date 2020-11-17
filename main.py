from Env import *
from Model import *
from Analyze import *

envTrain = Env("Data/select_train_data_30m.csv")
envTest = Env("Data/select_test_data_30m.csv")

print("Training...")
lossDQN, DQN = train(envTrain, n_neurons=18, nb_episodes=5, nb_steps=50, batch_size=10)
print("Done")

test(envTest, DQN)
