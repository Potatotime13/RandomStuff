import datetime
from trades import Executed_Trade
import random as rn

class Order():
    def __init__(self, volume_range:list, stock_list:list, avg_vol:dict, time_window_length:int) -> None:
        '''
        incoming order defined by its input parameters
        volume_range : percentage of daily volume
        stock_list : stocks which can be ordered
        time_window_length : time in hours
        '''
        self.MARKET_END = datetime.time(16,30)
        self.symbol = rn.choice(stock_list)
        self.volume = volume_range[0] + rn.random() * (volume_range[1]-volume_range[0]) * avg_vol[self.symbol]
        start_h = rn.randint(8,self.MARKET_END.hour-1-time_window_length)
        start_m = rn.randint(0,59)
        self.time_window = [datetime.time(start_h,start_m),datetime.time(start_h+time_window_length,start_m)]
        self.market_side = rn.choice(['buy','sell'])
        self.child_orders = []
        
        self.volume_left = self.volume
        self.vwap = None
    
    def execution(self, exec_trade:Executed_Trade):
        self.child_orders.append(exec_trade)
        self.volume_left -= exec_trade.volume
        if self.vwap == None:
            self.vwap = exec_trade.price
        else:
            self.vwap = self.vwap/(self.volume-self.volume_left) + exec_trade.price/exec_trade.volume
