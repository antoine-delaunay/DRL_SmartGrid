from Env import *
from Model import *
from Analyze import *

envTrain = Env("Data/select_train_data_30m.csv")
envTest = Env("Data/select_test_data_30m.csv")

print("Training...")
lossDQN, DQN = train(envTrain)
print("Done")

test(envTest, DQN)
