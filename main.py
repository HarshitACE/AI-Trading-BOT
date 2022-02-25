# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import abc
import threading
import time
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from alpaca_trade_api import REST


# Helper Class to manage the API connections

class AlpacaPaperSocket(REST):
    def __init__(self):
        super().__init__(
            key_id='PK5WZOSYJAVVAXZ8J5EP',
            secret_key='$4UY0veJ2ZuRZIcyxszDtFGLI5lSVTZUICZTqfmZL',
            base_url='https://paper-api.alpaca.markets'
        )


# Trading System Class with abstraction:
# - We are making the methods of this calss abstract so we can have different portfolio management systems based on our
# needs.
# - All the methods can be induvidually implemented by the app or requirement of the program
# - For example. System Loop for portfolio management would be very different to day trading systems


class TradingSystem(abc.ABC):

    def __init__(self, api, symbol, time_frame, system_id, system_label):
        # Connect to API
        # Connect to BrokenPipeError
        # Save the Fields to the class
        self.api = api
        self.symbol = symbol
        self.time_frame = time_frame
        self.system_id = system_id
        self.system_label = system_label
        thread = threading.Thread(target=self.system_loop)
        thread.start()

        @abc.abstractmethod
        def place_buy_order(self):
            pass

        @abc.abstractmethod
        def place_sell_order(self):
            pass

        @abc.abstractmethod
        def system_loop(self):
            pass


# Portfolio Management system Class

class PortfolioManagementSystem(TradingSystem):

    def __init__(self):
        super().__init__(AlpacaPaperSocket(), 'IBM', 604800, 1, 'AI_PM')

    def place_buy_order(self):
        self.api.submit_order(
            symbol='IBM',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day',
        )

    def place_sell_order(self):
        self.api.submit_order(
            symbol='IBM',
            qty=1,
            side='sell',
            type='market',
            time_in_force='day',
        )

    def system_loop(self):
        # Variables for weekly close
        this_weeks_close = 0
        last_weeks_close = 0
        delta = 0
        day_count = 0
        while (True):
            # Wait a day to request more data
            time.sleep(1440)
            # Request EoD data for IBM
            data_req = self.api.get_barset('IBM', timeframe='1D', limit=1).df
            # Construct dataframe to predict
            x = pd.DataFrame(
                data=[[
                    data_req['IBM']['close'][0]]], columns='Close'.split()
            )
            if (day_count == 7):
                day_count = 0
                last_weeks_close = this_weeks_close
                this_weeks_close = x['Close']
                delta = this_weeks_close - last_weeks_close

                # AI choosing to buy, sell, or hold
                if np.around(self.AI.network.predict([delta])) <= -.5:
                    self.place_sell_order()

                elif np.around(self.AI.network.predict([delta]) >= .5):
                    self.place_buy_order()

PortfolioManagementSystem()


# AI Model that we are going to train is going to be bare bones with the technique that we are using being
# by the dip and sell the rip (above and below a certain threshold)
# We will be building and annotating the data set based on the IBM market data and create a feature called signal which
# will yield the value in the set {-1, 0, 1} based on a threshold of change.

class AIPortfolioDevelopment:

    def __init__(self):
        # Read data and split it into dependent and indemependent variable:
        data = pd.read_csv('IBM.csv')
        X = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        # Train and Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Create the sequential
        network = Sequential()

        # Create a structure of the neural network
        network.add(Dense(1, input_shape=(1,), activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(1, activation='tanh'))

        # Compile the model
        network.compile(
            optimizer='rmsprop',
            loss='hinge',
            metrics=['accuracy']
        )

        # Train the model
        network.fit(X_train.values, y_train.values, epochs=100)

        # Evaluvate the predictions of the model
        y_pred = network.predict(X_test.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y_test, y_pred))

        # Save the structure to JSON
        model = network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model)

        # Save the weights to HDF5
        network.save_weights("weights.h5")


AIPortfolioDevelopment()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
