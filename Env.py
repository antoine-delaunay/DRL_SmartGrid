import pandas
import numpy as np
import datetime

# ACTIONS = ["charge", "discharge", "generator", "discharge + generator", "nothing"]
ACTIONS = ["charge", "discharge", "nothing"]
NB_ACTION = len(ACTIONS)
EPS = 0.5
GAMMA = 1.0


class State:
    def __init__(self):
        self.battery = 0.0
        self.panelProd = 0.0
        self.panelProd1 = 0.0
        self.panelProd2 = 0.0
        self.panelProd3 = 0.0
        self.panelProd4 = 0.0
        self.panelProd5 = 0.0

        self.consumption = 0.0
        self.consumption1 = 0.0
        self.consumption2 = 0.0
        self.consumption3 = 0.0
        self.consumption4 = 0.0
        self.consumption5 = 0.0

        self.price = 0.0
        self.daytime = 0.0
        self.row = 0

        self.charge = 0.0
        self.discharge = 0.0
        self.generate = 0.0
        self.trade = 0.0

    def toArray(self):
        return np.array(
            [
                self.battery,
                self.panelProd,
                self.panelProd1,
                self.panelProd2,
                self.panelProd3,
                self.panelProd4,
                self.panelProd5,
                self.consumption,
                self.consumption1,
                self.consumption2,
                self.consumption3,
                self.consumption4,
                self.consumption5,
                self.price,
                self.daytime,
            ]
        )


DIM_STATE = len(State().toArray())


class Env:
    def __init__(self, dataFile: str):
        # load data (csv)
        df = pandas.read_csv(dataFile, sep=";", header=0)

        self.data = df.values

        self.data[:, 1] = [
            (
                datetime.datetime.strptime(dateStr.split("+")[0], "%Y-%m-%d %H:%M:%S")
                - datetime.datetime.strptime(dateStr.split(" ")[0], "%Y-%m-%d")
            ).total_seconds()
            / (24 * 60 * 60)
            for dateStr in self.data[:, 1]
        ]

        # Prétraitement des données
        # TODO: transformer daytime en float
        self.panelProdMax = max(self.data[:, 5])
        self.consumptionMax = max(self.data[:, 4])
        self.priceMax = max(abs(self.data[:, 3]))

        self.data[:, 5] /= self.panelProdMax
        self.data[:, 4] /= self.consumptionMax
        self.data[:, 3] /= self.priceMax

        # Capacity of the battery and the generator
        self.initState()
        self.batteryCapacity = 10  # 60000.0 / self.panelProdMax
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

    def initState(self):
        self.currentState = State()
        self.currentState.row = np.random.randint(
            0, len(self.data)
        )  # Deuxième valeur à modifier en fonction du nombre de steps réalisés par épisode
        row = self.currentState.row
        self.currentState.daytime = self.data[row, 1]
        self.currentState.panelProd = self.data[row, 5]
        self.currentState.panelProd1 = self.data[row - 1, 5]
        self.currentState.panelProd2 = self.data[row - 2, 5]
        self.currentState.panelProd3 = self.data[row - 3, 5]
        self.currentState.panelProd4 = self.data[row - 4, 5]
        self.currentState.panelProd5 = self.data[row - 5, 5]

        self.currentState.price = self.data[row, 3]
        self.currentState.consumption = self.data[row, 4]
        self.currentState.consumption1 = self.data[row - 1, 4]
        self.currentState.consumption2 = self.data[row - 2, 4]
        self.currentState.consumption3 = self.data[row - 3, 4]
        self.currentState.consumption4 = self.data[row - 4, 4]
        self.currentState.consumption5 = self.data[row - 5, 4]

    def act(self, action):
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
                cost += self.currentState.charge * self.chargingCost

        elif action == "discharge":
            if self.diffProd < 0:
                self.currentState.discharge = max(
                    self.diffProd / self.dischargingYield, -self.currentState.battery
                )
                self.currentState.battery += self.currentState.discharge
                self.diffProd -= self.currentState.discharge * self.dischargingYield
                cost += abs(self.currentState.discharge * self.dischargingCost)

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
        self.currentState.daytime = self.data[row, 1]
        self.currentState.panelProd = self.data[row, 5]
        self.currentState.panelProd1 = self.data[row - 1, 5]
        self.currentState.panelProd2 = self.data[row - 2, 5]
        self.currentState.panelProd3 = self.data[row - 3, 5]
        self.currentState.panelProd4 = self.data[row - 4, 5]
        self.currentState.panelProd5 = self.data[row - 5, 5]

        self.currentState.price = self.data[row, 3]
        self.currentState.consumption = self.data[row, 4]
        self.currentState.consumption1 = self.data[row - 1, 4]
        self.currentState.consumption2 = self.data[row - 2, 4]
        self.currentState.consumption3 = self.data[row - 3, 4]
        self.currentState.consumption4 = self.data[row - 4, 4]
        self.currentState.consumption5 = self.data[row - 5, 4]

        self.currentState.row = row

        return -cost, self.currentState

    def getState(self):
        return self.currentState
