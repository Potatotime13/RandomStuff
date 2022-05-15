import numpy as np
import datetime
from trades import Executed_Trade

class Order():
    def __init__(self) -> None:
        '''
        incoming order defined by its input parameters
        volume_range : 
        stock_list : 
        time_window_length :
        '''
        self.volume = 0.1
        self.symbol = 'Daimler'
        self.time_window = [datetime.time(11,34),datetime.time(14,21)]
        self.market_side = 'buy'
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


    