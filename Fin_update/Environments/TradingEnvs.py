import numpy as np

class TradeEnv():
    '''
    DEFINE:
    -costs: costs per trade
    -seperator: percentage of time which is for the pretraining
    '''
    def __init__(self, costs, seperator) -> None:
        self.stocks = []
        self.time = []
        self.costs = costs
        self.separator = seperator

    def trading_simulation(self, model, tradeepochs, tradeset, tradesol, prices):
        '''
        -INPUTS:
        model: Keras Model
        epochs: training epochs [train, trade]
        trainset: pre shuffled and cutted timeseriesdata [input, labels]
        tradeset: linear timeseries data, in the future of training data, with overlap of size (window)
        prices: daily prices per stock [stock,day]

        -OUTPUT: trade period performance in decimal

        Longrun: pretraining optional, transaction cost, create simulation class(automated Data laoding and environment creation)
                    buy sell decision with additional heuristics
        '''
        def execute_trade(buy, sell, port, prices):
            port += buy - sell
            revenue = np.sum(sell * prices) - np.sum(buy * prices)
            revenue -= (np.sum(sell>0) + np.sum(buy>0)) * self.costs
            return revenue, port

        portfolio = np.zeros(prices.shape[0])
        money = 100000
        money_s = money

        # trading
        for i in range(tradeset.shape[0]):
            pred = model.predict(tradeset[i:i+1,:,:,:])[0,:]
            bu, se = model.decide(pred, money, prices[:,i], portfolio)
            rev, portfolio = execute_trade(bu, se, portfolio, prices[:,i])
            money += rev        
            if i > 0 and i % 20 == 0:
                model.fit(tradeset[i-20:i,:,:,:],tradesol[i-20:i,:], batch_size=4, epochs=tradeepochs)
        
        return (money + np.sum(prices[:,i] * portfolio))/money_s

    def load_data(self):
        with open('./Fin_update/Environments/test.npy', 'rb') as f:
            data = np.load(f)
        mid = int(data.shape[1] * self.separator)
        return data, mid, data.shape[0]