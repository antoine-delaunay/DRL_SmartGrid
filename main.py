from Env import Env
from Model import train, save, load
from Analyze import test

envTrain = Env("Data/select_train_data_30m.csv")
envTest = Env("Data/select_test_data_30m.csv")

print("Training...")
DQN = train(envTrain, n_neurons=5, nb_episodes=100, nb_steps=10, batch_size=100)
print("Done")

# save(DQN, "Models/test")
# DQN = load("Models/test")

test(envTest, DQN_model=DQN)

# test(envTest)
