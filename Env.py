import pandas
import numpy as np
import datetime

ACTIONS = np.array(["charge", "discharge", "trade"])
NB_STEPS_MEMORY = 10
BATTERY_CAPACITY = 10.0

# Test
f_c = 5
f_p = 10


class State:
    def __init__(self):
        self.battery = 0.0
        self.panelProd = 0.0
        self.consumption = 0.0
        self.price = 0.0
        self.row = 0

        self.charge = 0.0
        self.discharge = 0.0
        self.generate = 0.0
        self.trade = 0.0

        self.panelProdMemory = [0.0] * NB_STEPS_MEMORY
        self.consumptionMemory = [0.0] * NB_STEPS_MEMORY
        self.priceMemory = [0.0] * NB_STEPS_MEMORY

    def updateMemory(self):
        """ 
        The state memorize values of production, consumption and price over the last NB_STEPS_MEMORY steps.
        This function has to be called each time these parameters are updated.
        """
        self.panelProdMemory.pop(0)
        self.panelProdMemory.append(self.panelProd)
        self.consumptionMemory.pop(0)
        self.consumptionMemory.append(self.consumption)
        self.priceMemory.pop(0)
        self.priceMemory.append(self.price)

    def toArray(self):
        """ 
        Builds a np.array describing the essential values of the current state of the environment.
        The array generated in this function is expected to be used by the DQN algorithm.
    
        Returns: 
        np.array:  state of the environment
    
        """
        return np.array([self.battery] + self.panelProdMemory + self.consumptionMemory)


class Env:
    def __init__(self, dataFile: str):
        """
        Constants of the environment are defined here.
        Preprocessing of the data from dataFile.
    
        Parameters: 
        dataFile (str): a CSV file containing values of production, consumption and price over time 
    
        """
        # load data (csv)
        df = pandas.read_csv(dataFile, sep=",", header=0)

        self.data = df.values

        self.panelProdMax = max(self.data[:, 5]) / 1.5
        self.consumptionMax = max(self.data[:, 4])
        self.priceMax = max(abs(self.data[:, 3]))

        self.data[:, 5] /= self.panelProdMax
        self.data[:, 4] /= self.consumptionMax
        self.data[:, 3] /= self.priceMax

        # Capacity of the battery and the generator
        self.reset()
        self.batteryCapacity = BATTERY_CAPACITY
        self.generatorCapacity = (
            0.4  # Energie produite par le générateur en 5min 20000.0 / (12 * self.panelProdMax)
        )

        # CO2 price/emissions
        self.co2Price = 25.0 * 0.001  # price per ton of CO2 (mean price from the european market)
        self.co2Generator = 8 * 0.001  # kg of CO2 generated per kWh from the diesel generator
        self.co2Market = (
            0.3204  # kg of CO2 generated per kWh from the national power market (danish)
        )

        # Operational costs
        self.chargingCost = 0.0
        self.dischargingCost = 0.0
        # self.solarCost = 0.0
        self.generatorCost = 0.0  # 0.314 à 0.528 $/kWh

        # Yields
        self.chargingYield = 1.0
        self.dischargingYield = 1.0

    def reset(self, nb_step=0):
        """ 
        Reset randomly the current state of the environment.
    
        Parameters: 
        nb_step (int): maximum number of steps expected after the call to this function.
        This parameter is necessary since there are no terminal states and the data is not generated.
    
        """
        self.currentState = State()
        row = np.random.randint(NB_STEPS_MEMORY, len(self.data) - nb_step)
        for self.currentState.row in range(row - NB_STEPS_MEMORY, row + 1):
            # self.currentState.daytime = self.data[self.currentState.row, 1]
            self.currentState.price = 1.0
            # self.currentState.price = self.data[self.currentState.row, 3]
            self.currentState.consumption = self.data[self.currentState.row, 4]
            self.currentState.panelProd = self.data[self.currentState.row, 5]
            self.currentState.updateMemory()

        # Test
        # row = np.random.randint(NB_STEPS_MEMORY, 100 + NB_STEPS_MEMORY)

        # for self.currentState.row in range(row - NB_STEPS_MEMORY, row + 1):
        #     self.currentState.consumption = np.cos(2 * np.pi * self.currentState.row / f_c)
        #     self.currentState.panelProd = np.cos(2 * np.pi * self.currentState.row / f_p)
        #     self.currentState.updateMemory()

    def step(self, action):
        """ 
        Does the given action, and updates the environment accordingly.
    
        Parameters: 
        action (str): the action to do

        Returns: 
        reward (float):  reward associated to the current state and action
        
        state_updated (State): the new state of the environment

        """

        self.diffProd = self.currentState.panelProd - self.currentState.consumption
        cost = 0.0
        self.currentState.charge = 0.0
        self.currentState.discharge = 0.0
        self.currentState.generate = 0.0
        self.currentState.trade = 0.0

        if action == "charge":
            if self.diffProd > 0:
                self.currentState.charge = min(
                    self.diffProd,
                    (self.batteryCapacity - self.currentState.battery) / self.chargingYield,
                )
                self.currentState.battery += self.currentState.charge * self.chargingYield
                self.diffProd -= self.currentState.charge
                cost -= self.currentState.charge * self.currentState.price

        elif action == "discharge":
            if self.diffProd < 0:
                self.currentState.discharge = max(
                    self.diffProd / self.dischargingYield, -self.currentState.battery
                )
                self.currentState.battery += self.currentState.discharge
                self.diffProd -= self.currentState.discharge * self.dischargingYield

        # elif action == "generator":
        #     if self.diffProd < 0:
        #         self.currentState.generate = min(-self.diffProd, self.generatorCapacity)
        #         self.diffProd += self.currentState.generate
        #         cost += self.currentState.generate * self.generatorCost

        # elif action == "discharge + generator":
        #     if self.diffProd < 0:
        #         self.currentState.discharge = max(
        #             self.diffProd / self.dischargingYield, -self.currentState.battery
        #         )
        #         self.currentState.battery += self.currentState.discharge
        #         self.diffProd -= self.currentState.discharge * self.dischargingYield
        #         cost += abs(self.currentState.discharge * self.dischargingCost)

        #     if self.diffProd < 0:
        #         self.currentState.generate = min(-self.diffProd, self.generatorCapacity)
        #         self.diffProd += self.currentState.generate
        #         cost += self.currentState.generate * self.generatorCost

        self.currentState.trade = -self.diffProd

        if self.diffProd < 0:
            cost -= self.diffProd * self.currentState.price
        else:
            cost -= self.diffProd * self.currentState.price / 10

        row = self.currentState.row + 1
        if row >= len(self.data):
            self.currentState = None
        else:
            self.currentState.row = row
            self.currentState.daytime = self.data[row, 1]
            # self.currentState.price = self.data[row, 3]
            self.currentState.price = 1.0
            self.currentState.consumption = self.data[row, 4]
            self.currentState.panelProd = self.data[row, 5]

            # Test
            # self.currentState.consumption = np.cos(2 * np.pi * self.currentState.row / f_c)
            # self.currentState.panelProd = np.cos(2 * np.pi * self.currentState.row / f_p)

            self.currentState.updateMemory()

        return -cost, self.currentState

    def getState(self):
        """ 
        currentState (State): current state of the environment

        """
        return self.currentState
