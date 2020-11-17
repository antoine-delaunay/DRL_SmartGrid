from Env import *
from Model import *
from Analyse import *

envTrain = Env("select_train_data_30m.csv")

print("Training...")
lossDQN, DQN = train(envTrain)
print("Done")

test(envTest, DQN)
