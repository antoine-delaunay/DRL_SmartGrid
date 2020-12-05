import pandas
import numpy as np
import datetime

ACTIONS = np.array(["charge", "discharge", "trade"])
NB_STEPS_MEMORY = 10
BATTERY_CAPACITY = 2.0


class State:
    def __init__(self):
        self.battery = 0.0
        self.panelProd = 0.0
        self.consumption = 0.0
        self.price = 0.0
        # self.daytime = 0.0
        self.row = 0

        self.charge = 0.0
        self.discharge = 0.0
        self.generate = 0.0
        self.trade = 0.0

        self.panelProdMemory = [0.0] * NB_STEPS_MEMORY
        self.consumptionMemory = [0.0] * NB_STEPS_MEMORY
        self.priceMemory = [0.0] * NB_STEPS_MEMORY

    def updateMemory(self):
        self.panelProdMemory.pop(0)
        self.panelProdMemory.append(self.panelProd)
        self.consumptionMemory.pop(0)
        self.consumptionMemory.append(self.consumption)
        self.priceMemory.pop(0)
        self.priceMemory.append(self.price)

    def toArray(self):
        return np.array([self.battery] + self.panelProdMemory + self.consumptionMemory)


class Env:
    def __init__(self, dataFile: str):
        # load data (csv)
        df = pandas.read_csv(dataFile, sep=",", header=0)

        self.data = df.values

        # Prétraitement des données
        # self.data[:, 1] = [
        #     (
        #         datetime.datetime.strptime(dateStr.split("+")[0], "%Y-%m-%d %H:%M:%S")
        #         - datetime.datetime.strptime(dateStr.split(" ")[0], "%Y-%m-%d")
        #     ).total_seconds()
        #     / (24 * 60 * 60)
        #     for dateStr in self.data[:, 1]
        # ]

        self.panelProdMax = max(self.data[:, 5])
        self.consumptionMax = max(self.data[:, 4])
        self.priceMax = max(abs(self.data[:, 3]))

        self.data[:, 5] /= self.panelProdMax
        self.data[:, 4] /= self.consumptionMax
        self.data[:, 3] /= self.priceMax

        # Capacity of the battery and the generator
        self.reset()
        self.batteryCapacity = BATTERY_CAPACITY  # 60000.0 / self.panelProdMax
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
        self.currentState = State()
        # row = np.random.randint(NB_STEPS_MEMORY, len(self.data) - nb_step)
        # for self.currentState.row in range(row - NB_STEPS_MEMORY, row + 1):
        #     # self.currentState.daytime = self.data[self.currentState.row, 1]
        #     self.currentState.price = 1.0
        #     # self.currentState.price = self.data[self.currentState.row, 3]
        #     self.currentState.consumption = self.data[self.currentState.row, 4]
        #     self.currentState.panelProd = self.data[self.currentState.row, 5]
        #     self.currentState.updateMemory()

        # Test
        f_c = 5
        f_p = 10
        row = np.random.randint(NB_STEPS_MEMORY, 100 + NB_STEPS_MEMORY)

        for self.currentState.row in range(row - NB_STEPS_MEMORY, row + 1):
            self.currentState.consumption = np.cos(2 * np.pi * self.currentState.row / f_c)
            self.currentState.panelProd = np.cos(2 * np.pi * self.currentState.row / f_p)
            self.currentState.updateMemory()

    def step(self, action):
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
                # cost += self.currentState.charge * self.chargingCost
                if self.currentState.charge > 1e-5:
                    cost = -1.0
                else:
                    cost = 0.0
            else:
                cost = 1.0

        elif action == "discharge":
            if self.diffProd < 0:
                self.currentState.discharge = max(
                    self.diffProd / self.dischargingYield, -self.currentState.battery
                )
                self.currentState.battery += self.currentState.discharge
                self.diffProd -= self.currentState.discharge * self.dischargingYield
                # cost += abs(self.currentState.discharge * self.dischargingCost)
                if self.currentState.discharge < -1e-5:
                    if self.diffProd < -1e-5:
                        cost = 0.25
                    else:
                        cost = -1.0
                else:
                    cost = 1.0
            else:
                cost = 0.0

        elif action == "trade":
            if self.diffProd < 0:
                cost = 0.5
            else:
                cost = 0.0

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

        # cost -= 3 * self.currentState.battery / self.batteryCapacity

        # if self.diffProd < 0:
        #     cost -= self.diffProd * self.currentState.price
        # else:
        #     cost -= self.diffProd * self.currentState.price / 10

        # if self.diffProd < -1e-3:
        #     cost = 1.0

        row = self.currentState.row + 1
        if row >= len(self.data):
            self.currentState = None
        else:
            self.currentState.row = row
            self.currentState.daytime = self.data[row, 1]
            # self.currentState.price = self.data[row, 3]
            self.currentState.price = 1.0
            # self.currentState.consumption = self.data[row, 4]
            # self.currentState.panelProd = self.data[row, 5]

            # Test
            f_c = 5
            f_p = 10
            self.currentState.consumption = np.cos(2 * np.pi * self.currentState.row / f_c)
            self.currentState.panelProd = np.cos(2 * np.pi * self.currentState.row / f_p)

            self.currentState.updateMemory()

        return -cost, self.currentState

    def getState(self):
        return self.currentState
