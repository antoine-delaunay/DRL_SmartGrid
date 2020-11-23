from Env import Env
from Model import train, save, load
from Analyze import test

envTrain = Env("Data/select_train_data_30m.csv")
envTest = Env("Data/select_test_data_30m.csv")

model1 = "Res_NN_10_num1"

print("Training...")
DQN = train(envTrain, n_neurons=10, nb_episodes=100, nb_steps=100, batch_size=100, model_name="Res_NN_10_num1", recup_model=True)
print("Done")

# save(DQN, "Models/test")
# DQN = load("Models/test")

test(envTest, DQN_model=DQN)

# test(envTest)
